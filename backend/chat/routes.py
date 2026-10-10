"""
对话路由 — SSE流式对话，接入LangGraph四阶段流水线
审核修正：
- 对话前先做RAG检索，将知识上下文注入state
- 按工作区模式注入对应教学提示（见 MODE_PROMPTS），模式集合与前端选择器一致
- SSE流式输出（多智能体进度可视化）
- 命令系统：/test /plan /config /start /continue
- 学习闭环：错题自动入本，自适应难度调整
"""
import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.graph import run_tutor_graph, run_tutor_graph_stream
from backend.agents.prompts import ANSWER_PROMPT, HINT_LEVEL_PROMPTS, HINT_SYSTEM_PROMPT, MODE_PROMPTS
from backend.audit import record_api_usage
from backend.compression.engine import compression_engine
from backend.content_safety.filter import content_safety_filter
from backend.database import get_db, get_write_db
from backend.dependencies import get_current_user
from backend.evolution import dialectic_profiler
from backend.evolution import synthesis as evolution_synthesis
from backend.memory import summary_engine
from backend.models import (
    Conversation,
    ConversationMode,
    Message,
    User,
    UserProfile,
    WrongQuestion,
)
from backend.personas import persona_engine
from backend.rag.knowledge_graph import knowledge_graph
from backend.rag.retriever import retriever
from backend.rag.web_search import web_searcher
from backend.tutor.evidence import (
    collect_learning_evidence,
    persist_graph_result,
    record_learning_session,
)
from backend.tutor.wrong_diagnoser import wrong_diagnoser

router = APIRouter()

# Conversation.mode 列只接受 ConversationMode 枚举值；intent 可能是 chat/deep_solve/research/math/agent 等非枚举值
_VALID_MODES = {m.value for m in ConversationMode}


def _conversation_mode(mode: str | None) -> str:
    """工作区模式 → 会话枚举：非枚举值（chat/deep_solve/research/math…）统一归入 normal"""
    return mode if mode in _VALID_MODES else "normal"


async def _attach_review_flag(db, user_id: int, raw_content: str, ai_msg: Message):
    """审批开启时的AI输出事后审核：命中敏感内容则进审批队列（落库为打码版，教师通过后恢复全文）"""
    try:
        from backend.settings import load_settings
        if not load_settings().get("safety", {}).get("approval_required"):
            return
        hit, reason = content_safety_filter.needs_review(raw_content)
        if not hit:
            return
        await db.flush()
        from backend.models import ApprovalRecord
        db.add(ApprovalRecord(
            user_id=user_id, message_id=ai_msg.id, content=raw_content,
            content_type="ai_response", reason=reason,
        ))
        logger.info(f"AI输出进入审批队列: user={user_id} reason={reason}")
    except Exception as e:
        logger.warning(f"审批标记失败: {e}")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    mode: str = Field(default="chat", pattern="^(chat|agent|code_act|deep_solve|quiz|research|math|auto|normal|socratic|essay)$")
    knowledge_base_id: int | None = None
    skill_id: str | None = None
    # A3 计划确认：携带时进入 plan_execute Phase 2（从检查点恢复执行/取消/编辑后执行）
    plan_confirm: dict | None = None
    # 深度思考开关：控制是否开启thinking模式（GLM-4.5+支持）
    deep_thinking: bool = True
    # 联网搜索开关：控制是否进行联网搜索补充知识
    web_search: bool = False


def _resolve_skill(req_skill_id: str | None, message: str) -> tuple[str, dict[str, Any] | None]:
    """解析生效技能 → (注入提示词的约束文本, 技能信息)

    显式选择的技能必须存在，静默降级会让用户以为技能生效了；
    未显式选择时按 SKILL.md 的触发词自动匹配。
    """
    from backend.skills.engine import skills_engine

    if req_skill_id:
        skill = skills_engine.get_skill(req_skill_id)
        if not skill:
            raise HTTPException(status_code=404, detail=f"技能 '{req_skill_id}' 不存在")
        auto = False
    else:
        skill = skills_engine.match_skill(message)
        auto = True

    if not skill:
        return "", None

    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(skill.steps, 1))
    parts = [f"【启用技能：{skill.title}】", skill.description, f"请按以下步骤执行：\n{steps}"]
    # 执行合约：必需工具（agent 必须调用，不可跳过）
    if skill.required_tools:
        tools_str = ", ".join(skill.required_tools)
        parts.append(f"【执行合约】本技能必须使用以下工具，不可跳过：{tools_str}")
    # 执行合约：自检清单（执行后必须逐项确认）
    if skill.validation:
        parts.append(f"【执行后自检】\n{skill.validation}")
    prompt = "\n\n".join(parts)
    return prompt, {"id": skill.id, "title": skill.title, "auto": auto}


class ConversationResponse(BaseModel):
    id: str
    title: str
    mode: str
    created_at: str
    updated_at: str


# ===== QA 问答缓存（仅 chat 模式） =====
# 重复提问秒回并省 token；TTL 内命中直接复用 AI 回复，消息照常落库。
# 排除：命令/苏格拉底（阶段状态推进）/出题/研究/挂接知识库或技能的请求——这些需要实时性。
_QA_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_QA_CACHE_TTL = 7 * 86400
_QA_CACHE_MAX = 200


def _qa_cache_key(user_id: int, message: str, mode: str, model_name: str) -> str:
    raw = f"{user_id}|{message.strip()}|{mode}|{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _qa_cache_get(key: str) -> dict[str, Any] | None:
    hit = _QA_CACHE.get(key)
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts > _QA_CACHE_TTL:
        _QA_CACHE.pop(key, None)
        return None
    # 历史坏缓存自愈：此前失败的"不可用"回复被误写进缓存，
    # 读到时按未命中处理并清掉，避免 API 恢复后仍秒回失败。
    if _qa_reply_is_error(str(payload.get("reply", ""))):
        _QA_CACHE.pop(key, None)
        logger.warning(f"QA缓存清除失败兜底条目: {key[:8]}")
        return None
    _QA_CACHE.move_to_end(key)
    return payload


def _qa_cache_put(key: str, payload: dict[str, Any]) -> None:
    _QA_CACHE[key] = (time.time(), payload)
    _QA_CACHE.move_to_end(key)
    while len(_QA_CACHE) > _QA_CACHE_MAX:
        _QA_CACHE.popitem(last=False)


def _qa_reply_is_error(reply: str) -> bool:
    """失败回复判定：LLM 全通道不可用时返回的兜底文案/空回复不得进缓存

    历史上失败兜底（"抱歉，AI服务暂时不可用"）也被缓存 7 天，
    API 恢复后同样的问题仍秒回失败——这是"项目不稳定"体感的典型来源之一。
    """
    if not reply or not reply.strip():
        return True
    return reply.strip().startswith("抱歉，AI服务暂时不可用") or reply.strip().startswith("抱歉，我暂时无法回答")


def _qa_cache_eligible(req: "ChatRequest", cmd: dict | None) -> bool:
    return (
        cmd is None
        and req.mode == "chat"
        and not req.knowledge_base_id
        and not req.skill_id
        and not req.message.lstrip().startswith("/")
    )


# 命令解析
COMMAND_PATTERN = re.compile(r'^/(\w+)(?:\s+(.*))?$')


def parse_command(message: str) -> dict[str, Any] | None:
    """解析斜杠命令"""
    match = COMMAND_PATTERN.match(message.strip())
    if not match:
        return None
    cmd = match.group(1).lower()
    args = match.group(2) or ""
    return {"command": cmd, "args": args, "raw": message}


async def get_user_profile(db: AsyncSession, user_id: int) -> UserProfile | None:
    """获取用户配置（不存在则创建默认）"""
    from sqlalchemy import select
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        await db.flush()
    return profile


def build_profile_prompt(profile: UserProfile) -> str:
    """根据用户配置生成个性化prompt片段"""
    depth_labels = {
        "elementary": "小学水平，用最简单的语言和例子",
        "middle_school": "初中水平，系统讲解基础概念",
        "high_school": "高中水平，深入理解并适当拓展",
        "college_prep": "大学预科水平，注重思维方法",
        "undergraduate": "本科水平，专业系统讲解",
        "graduate": "研究生水平，注重研究方法",
        "masters": "硕士水平，课题导向深入研究",
        "phd_candidate": "博士候选人水平，前沿探索",
        "postdoc": "博士后水平，跨学科创新",
        "phd": "博士水平，最高学术标准",
    }
    style_labels = {
        "visual": "多用图表、示意图、可视化比喻",
        "verbal": "注重文字阐述、逻辑论证",
        "active": "多设计互动练习、动手实践",
        "intuitive": "注重概念直觉、整体把握",
        "reflective": "多留思考空间、引导反思总结",
        "global": "宏观视角、系统思维、联系整体",
    }
    comm_labels = {
        "format": "结构化输出，使用标题、列表、表格",
        "textbook": "教科书式严谨讲解，定义+定理+例题",
        "layman": "通俗大白话，生活化比喻，避免专业术语",
        "story_telling": "通过故事和案例讲解知识点",
        "socratic": "苏格拉底式，通过提问引导思考",
    }
    tone_labels = {
        "encouraging": "鼓励型语气，积极正向，给予信心",
        "neutral": "中立客观，就事论事",
        "informative": "信息密集，干货满满",
        "friendly": "亲切友好，像朋友聊天",
        "humorous": "幽默风趣，寓教于乐",
    }
    reason_labels = {
        "deductive": "演绎推理，从一般原理推导出具体结论",
        "inductive": "归纳推理，从具体例子总结出一般规律",
        "abductive": "溯因推理，从结果反推可能的原因",
        "analogical": "类比推理，通过相似性建立联系",
        "causal": "因果推理，分析因果关系，追根溯源",
    }

    return f"""
【学生个性化配置】
- 学习深度：{depth_labels.get(profile.learning_depth.value, '高中水平')}
- 学习风格：{style_labels.get(profile.learning_style.value, '视觉型')}
- 沟通方式：{comm_labels.get(profile.communication_style.value, '通俗大白话')}
- 语气风格：{tone_labels.get(profile.tone_style.value, '鼓励型')}
- 推理框架：{reason_labels.get(profile.reasoning_framework.value, '因果推理')}
- 当前难度系数：{profile.current_difficulty:.2f}（0=最简单，1=最难）
请严格按照以上配置调整你的教学方式和内容深度。
""".strip()


async def _build_history(db: AsyncSession, conversation_id: str, user_id: int) -> list[dict[str, str]]:
    """构建上下文历史：取**最近** N 条，长会话再用压缩摘要替代其中的早期轮次

    两个历史缺陷都藏在这一行里：
    ① `.order_by(created_at).limit(n)` 是升序取前 N —— 拿到的是**最旧** N 条。
       十轮之后模型完全看不到当前讨论，用户体感"AI 忘了我刚说什么"，
       而且每轮都白花一次带错误上下文的 LLM 调用。
    ② 压缩引擎的阈值（15 条）作用在被取错的那批最旧消息上 —— 它一直在摘要
       八竿子打不着的远古内容，"上下文压缩"这个特性实际上没在干它名字该干的活。
    created_at 精度到秒级时会并列，用自增主键 id 兜底保证顺序确定。
    """
    from sqlalchemy import select
    result = await db.execute(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(compression_engine.max_history)
    )
    history = [{"role": m.role, "content": m.content} for m in reversed(result.scalars().all())]

    if not compression_engine.should_compress(history):
        return history

    try:
        packed = await compression_engine.compress(history, user_id)
        if packed.get("compressed"):
            logger.info(
                f"上下文压缩: {packed.get('original_length')}条 → {packed.get('compressed_length')}条"
                f"（摘要{len(packed.get('summary', {}).get('summary', ''))}字）"
            )
            return packed["history"]
    except Exception as e:
        logger.warning(f"上下文压缩失败，退回原始窗口: {e}")
    return history


@router.post("/send")
async def send_message(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_write_db),
):
    """发送消息 — 非流式版本，返回完整回复"""
    start_time = time.time()

    from backend.settings.routes import bind_user_model
    # 接住返回值：QA 缓存 key 需要与 stream 路径用同一 model_name，否则两路缓存互不相通
    request_model_cfg = await bind_user_model(db, current_user.id)

    ok, reason = content_safety_filter.check_input(req.message)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    # 获取或创建会话
    conversation = None
    if req.conversation_id:
        from sqlalchemy import select
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == req.conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
        conversation = result.scalar_one_or_none()

    if not conversation:
        import uuid
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            title=req.message[:30],
            mode=_conversation_mode(req.mode),
        )
        db.add(conversation)
        await db.flush()

    # 保存用户消息
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=req.message,
    )
    db.add(user_msg)

    # 获取对话历史（超长会话走压缩，不硬截断）
    history = await _build_history(db, conversation.id, current_user.id)

    # RAG检索（获取知识上下文 + 可溯源引用）；三态：ok|empty|unavailable，故障不伪装成"没查到"
    knowledge_context, knowledge_sources, retrieval_status = "", [], "unavailable"
    try:
        retrieved = await retriever.retrieve(req.message, req.knowledge_base_id)
        knowledge_context = retrieved.get("context", "")
        knowledge_sources = retrieved.get("sources", [])
        retrieval_status = retrieved.get("retrieval_status", "ok")
    except Exception as e:
        logger.warning(f"RAG检索不可用: {e}")

    # ===== 注入人格、自进化、记忆、模式（与stream_message保持一致） =====
    profile = await get_user_profile(db, current_user.id)
    profile_prompt = build_profile_prompt(profile)

    from backend.settings.user_store import get_user_setting
    persona_id = await get_user_setting(db, current_user.id, "persona") or "patient-teacher"
    persona_prompt = persona_engine.get_system_prompt(persona_id)
    if not persona_prompt:
        persona_id = "patient-teacher"
        persona_prompt = persona_engine.get_system_prompt(persona_id)
    persona_behavior = persona_engine.get_behavior_params(persona_id)

    evolution_data = dialectic_profiler.get_profile(current_user.id)
    evolution_profile = evolution_data.get("synthesis", "")
    evolution_style = evolution_data.get("learning_style", "")
    evolution_level = evolution_data.get("knowledge_level", "")

    memory_prompt = summary_engine.build_memory_prompt(current_user.id)

    # 学情证据：掌握度与错误类型分布注入 Profiler（此前无人写入，画像恒按"新用户"分析）
    learning_evidence = await collect_learning_evidence(db, current_user.id)

    # MODE_PROMPTS 已收敛到 backend/agents/prompts.py（消除 send/stream 漂移）
    mode_prompt = MODE_PROMPTS.get(req.mode, MODE_PROMPTS["chat"])
    skill_prompt, active_skill = _resolve_skill(req.skill_id, req.message)
    if skill_prompt:
        mode_prompt = f"{mode_prompt}\n\n{skill_prompt}"

    extra_state = {
        "profile_prompt": profile_prompt,
        "persona_id": persona_id,
        "persona_prompt": persona_prompt,
        "persona_behavior": persona_behavior,
        "evolution_profile": evolution_profile,
        "evolution_style": evolution_style,
        "evolution_level": evolution_level,
        "memory_prompt": memory_prompt,
        "mode_prompt": mode_prompt,
        "sources": knowledge_sources,
        **learning_evidence,
    }

    # deep_solve 需要计划确认交互，仅 /stream 支持；/send 明确报错而非静默降级为 react 直答
    if req.mode == "deep_solve" and not req.plan_confirm:
        raise HTTPException(status_code=400, detail="计划式解题仅支持流式对话（需计划确认交互），请在流式模式下使用。")

    # ===== Agent 系模式：与 /stream 同源的可插拔 Loop 路由（修复薄弱点⑧） =====
    # 此前 /send 的 agent 模式静默降级 normal_chat（图路由未命中），双入口行为不一致。
    # 非流式端点收集 loop 事件取最终 answer；plan_execute 的 Phase 1 会暂停等确认，
    # /send 场景无法交互确认 → deep_solve 在 /send 中回落 react loop 直答。
    if req.mode in ("agent", "code_act", "deep_solve", "math") and not req.plan_confirm:
        from backend.agents.loops import get_loop, route_loop
        system_context = f"{persona_prompt}\n\n{evolution_profile}\n\n{memory_prompt}\n\n{mode_prompt}"
        loop_name = "react" if req.mode == "deep_solve" else route_loop(req.mode)
        loop = get_loop(loop_name)
        final_content, final_tokens, final_model, agent_sources = "", 0, None, []
        async for ev in loop.run(
            message=req.message, history=history, system_context=system_context,
            user_id=current_user.id, conversation_id=conversation.id,
        ):
            if ev.get("type") == "answer":
                final_content = ev.get("content", "")
                final_tokens = ev.get("tokens_used", 0) or 0
                final_model = ev.get("model_used")
            elif ev.get("type") == "observation" and ev.get("tool") == "knowledge_search" and ev.get("ok"):
                seen = {s.get("chunk_id") for s in agent_sources}
                for hit in (ev.get("data") or {}).get("results", []):
                    if hit.get("chunk_id") not in seen:
                        agent_sources.append(hit)
                        seen.add(hit.get("chunk_id"))
        raw_reply = final_content
        final_content = content_safety_filter.filter_output(final_content)
        ai_msg = Message(
            conversation_id=conversation.id, role="assistant", content=final_content,
            sources=agent_sources or None, latency_ms=int((time.time() - start_time) * 1000),
            model_used=final_model, tokens_used=final_tokens,
        )
        db.add(ai_msg)
        await _attach_review_flag(db, current_user.id, raw_reply, ai_msg)
        conversation.updated_at = datetime.utcnow()
        await db.commit()
        return {
            "conversation_id": conversation.id, "message_id": ai_msg.id,
            "content": final_content, "sources": agent_sources,
            "model_used": final_model, "tokens_used": final_tokens,
        }

    # 运行LangGraph四阶段流水线（QA 缓存命中时跳过——重复提问秒回并省 token）
    _qa_model = (request_model_cfg or {}).get("model_name", "") or ""
    _qa_key = _qa_cache_key(current_user.id, req.message, req.mode, _qa_model) if _qa_cache_eligible(req, None) else None
    _qa_cached = _qa_cache_get(_qa_key) if _qa_key else None
    if _qa_cached:
        final_state = {
            "full_content": _qa_cached["reply"],
            "sources": _qa_cached.get("sources", []),
            "verification": _qa_cached.get("verification", {}),
            "model_used": _qa_cached.get("model_used"),
            "tokens_used": 0,
            "cached": True,
        }
        logger.info("QA缓存命中（send）")
    else:
        final_state = await run_tutor_graph(
            user_id=current_user.id,
            conversation_id=conversation.id,
            message=req.message,
            mode=req.mode,
            history=history,
            knowledge_context=knowledge_context,
            deep_thinking=req.deep_thinking,
            extra_state=extra_state,
        )

    # 保存AI回复
    raw_reply = final_state.get("full_content", "抱歉，我暂时无法回答。")
    ai_content = content_safety_filter.filter_output(raw_reply)
    sources = final_state.get("sources", [])
    verification = final_state.get("verification", {})
    if _qa_key and not _qa_cached and ai_content and not _qa_reply_is_error(ai_content):
        _qa_cache_put(_qa_key, {
            "reply": ai_content,
            "sources": sources,
            "verification": verification,
            "model_used": final_state.get("model_used"),
        })

    ai_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=ai_content,
        sources=sources if sources else None,
        verification=verification if verification else None,
        model_used=final_state.get("model_used"),
        tokens_used=final_state.get("tokens_used", 0),
        latency_ms=int((time.time() - start_time) * 1000),
    )
    db.add(ai_msg)
    await _attach_review_flag(db, current_user.id, raw_reply, ai_msg)

    # 更新会话
    conversation.updated_at = datetime.utcnow()
    if final_state.get("intent") and final_state["intent"] in _VALID_MODES:
        conversation.mode = final_state["intent"]

    # 掌握度回灌：对话即改变BKT（与AI回复同一事务）
    mastery_update = await persist_graph_result(db, current_user.id, final_state)
    await record_learning_session(
        db, current_user.id, conversation_id=conversation.id,
        mode=req.mode, started_ts=start_time, mastery_update=mastery_update,
    )

    await db.commit()

    # LLM 用量落库（api_key_usage：Key 消耗可追溯）
    _model = final_state.get("model_used")
    if _model:
        try:
            _total = final_state.get("tokens_used", 0) or 0
            _prompt = final_state.get("prompt_tokens", 0) or 0
            await record_api_usage(
                db, current_user.id, _model,
                prompt_tokens=_prompt,
                completion_tokens=max(_total - _prompt, 0),
                endpoint="/api/chat/send",
            )
        except Exception as e:
            logger.warning(f"API用量记录失败: {e}")

    # ===== Hermes自进化 + Summary生成（与stream_message保持一致） =====
    # fire-and-forget：这两个后置步骤各含一次 LLM 调用，串行 await 会拖慢用户拿到响应；
    # 后台自持短事务会话，失败只记日志不影响主流程
    _spawn_post_response_tasks(current_user.id, req.message, ai_content, req.mode, verification)

    return {
        "conversation_id": conversation.id,
        "reply": ai_content,
        "sources": sources,
        "retrieval_status": retrieval_status,
        "skill": active_skill,
        "verification": verification,
        "mastery_update": mastery_update,
        "intent": final_state.get("intent"),
        "socratic_stage": final_state.get("socratic_stage"),
        "iteration": final_state.get("iteration_count", 0),
        "tokens_used": final_state.get("tokens_used", 0),
        "prompt_tokens": final_state.get("prompt_tokens", 0),
        "model_used": final_state.get("model_used"),
        "latency_ms": ai_msg.latency_ms,
    }


@router.post("/stream")
async def stream_message(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_write_db),
):
    """SSE流式对话 — 推送多智能体执行进度和最终回复"""
    start_time = time.time()

    from backend.settings.routes import bind_user_model
    request_model_cfg = await bind_user_model(db, current_user.id)

    ok, reason = content_safety_filter.check_input(req.message)
    if not ok:
        async def blocked():
            yield f"data: {json.dumps({'type': 'error', 'message': reason}, ensure_ascii=False)}\n\n"
        return StreamingResponse(blocked(), media_type="text/event-stream")

    # 命令处理
    cmd = parse_command(req.message)
    if cmd:
        return StreamingResponse(
            handle_command(cmd, current_user, db, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 获取或创建会话
    from sqlalchemy import select
    conversation = None
    if req.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == req.conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
        conversation = result.scalar_one_or_none()

    if not conversation:
        import uuid
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            title=req.message[:30],
            mode=_conversation_mode(req.mode),
        )
        db.add(conversation)
        await db.flush()

    # 保存用户消息（plan_confirm 恢复轮不再重复落库用户消息）
    if not req.plan_confirm:
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=req.message,
        )
        db.add(user_msg)
    # 必须在返回 StreamingResponse 前落库：依赖会话会在流开始前回滚未提交事务
    await db.commit()

    # 获取对话历史（超长会话走压缩，不硬截断）
    history = await _build_history(db, conversation.id, current_user.id)

    # ===== QA 缓存（仅 chat 模式，命中跳过 RAG/LangGraph 全链路）=====
    # 命令已在上文处理；苏格拉底/出题/研究/挂接知识库或技能的请求需实时性，不缓存
    _qa_key = None
    _qa_cached = None
    if _qa_cache_eligible(req, cmd):
        try:
            _model_for_cache = (request_model_cfg or {}).get("model_name", "") or ""
        except Exception:
            _model_for_cache = ""
        _qa_key = _qa_cache_key(current_user.id, req.message, req.mode, _model_for_cache)
        _qa_cached = _qa_cache_get(_qa_key)

    if _qa_cached:
        async def cached_event_generator():
            cached_reply = _qa_cached["reply"]
            ai_msg = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=cached_reply,
                sources=_qa_cached.get("sources") or None,
                model_used=_qa_cached.get("model_used"),
                tokens_used=0,
                latency_ms=int((time.time() - start_time) * 1000),
            )
            db.add(ai_msg)
            conversation.updated_at = datetime.utcnow()
            await db.commit()
            logger.info("QA缓存命中（stream）")
            yield f"data: {json.dumps({'type': 'complete', 'full_content': cached_reply, 'sources': _qa_cached.get('sources', []), 'verification': _qa_cached.get('verification', {}), 'tokens_used': 0, 'prompt_tokens': 0, 'model_used': _qa_cached.get('model_used'), 'cached': True}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation.id}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            cached_event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 获取用户个性化配置
    profile = await get_user_profile(db, current_user.id)
    profile_prompt = build_profile_prompt(profile)

    # ===== 人格系统：获取用户选择的人格，注入System Prompt =====
    from backend.settings.user_store import get_user_setting
    persona_id = await get_user_setting(db, current_user.id, "persona") or "patient-teacher"
    persona_prompt = persona_engine.get_system_prompt(persona_id)
    if not persona_prompt:
        persona_id = "patient-teacher"
        persona_prompt = persona_engine.get_system_prompt(persona_id)
    persona_behavior = persona_engine.get_behavior_params(persona_id)
    logger.info(f"使用人格: {persona_id}")

    # ===== Hermes自进化：获取用户画像，注入上下文 =====
    evolution_data = dialectic_profiler.get_profile(current_user.id)
    evolution_profile = evolution_data.get("synthesis", "")
    evolution_style = evolution_data.get("learning_style", "")
    evolution_level = evolution_data.get("knowledge_level", "")
    if evolution_profile:
        logger.info(f"Hermes画像版本: v{evolution_data.get('version', 0)}")

    # ===== 双维度记忆：获取Summary学习进度摘要，注入上下文 =====
    memory_prompt = summary_engine.build_memory_prompt(current_user.id)
    if memory_prompt:
        logger.info(f"注入学习记忆: {len(memory_prompt)}字")

    # ===== 统一工作区5模式：根据模式生成专用指令 =====
    # MODE_PROMPTS 已收敛到 backend/agents/prompts.py（消除 send/stream 漂移）
    mode_prompt = MODE_PROMPTS.get(req.mode, MODE_PROMPTS["chat"])
    skill_prompt, active_skill = _resolve_skill(req.skill_id, req.message)
    if skill_prompt:
        mode_prompt = f"{mode_prompt}\n\n{skill_prompt}"
    logger.info(f"工作区模式: {req.mode}, 生效技能: {active_skill['id'] if active_skill else '无'}")

    # RAG检索（知识上下文 + 可溯源引用）；三态如实传播
    knowledge_context, knowledge_sources, retrieval_status = "", [], "unavailable"
    try:
        retrieved = await retriever.retrieve(req.message, req.knowledge_base_id)
        knowledge_context = retrieved.get("context", "")
        knowledge_sources = retrieved.get("sources", [])
        retrieval_status = retrieved.get("retrieval_status", "ok")
    except Exception as e:
        logger.warning(f"RAG检索不可用: {e}")

    # 联网搜索：当用户开启web_search时，补充最新信息
    web_search_results = []
    if req.web_search:
        try:
            web_search_results = await web_searcher.search(req.message, num_results=5)
            if web_search_results:
                web_context = web_searcher.format_as_context(web_search_results)
                knowledge_context = f"{knowledge_context}\n\n{web_context}" if knowledge_context else web_context
                # 将搜索结果也加入sources
                for r in web_search_results:
                    knowledge_sources.append({
                        "title": r["title"],
                        "url": r["url"],
                        "snippet": r["snippet"],
                        "source_type": "web",
                        "relevance": 0.8
                    })
                retrieval_status = "ok"
                logger.info(f"联网搜索返回 {len(web_search_results)} 条结果")
        except Exception as e:
            logger.warning(f"联网搜索失败: {e}")

    # 学情证据：掌握度与错误类型分布注入 Profiler（此前无人写入，画像恒按"新用户"分析）
    learning_evidence = await collect_learning_evidence(db, current_user.id)

    # 推送会话信息
    async def event_generator():
        # 流式响应的生成器运行在另一个 task 上下文，需重新绑定本人模型配置
        from backend.settings.model_ctx import set_model_override
        set_model_override(request_model_cfg)
        yield f"data: {json.dumps({'type': 'session', 'conversation_id': conversation.id}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'skill', 'skill': active_skill}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'retrieval', 'status': retrieval_status}, ensure_ascii=False)}\n\n"
        # 联网搜索结果：把引用来源传给前端展示（仿 DeepSeek 搜索引用）
        if req.web_search and web_search_results:
            citations = web_searcher.get_citations(web_search_results)
            yield f"data: {json.dumps({'type': 'web_search', 'results': citations, 'count': len(citations)}, ensure_ascii=False)}\n\n"

        final_content = ""
        final_sources = []
        final_verification = {}
        final_tokens = 0
        final_prompt_tokens = 0
        final_model = None
        bkt_pending = None

        # ===== Agent 系模式：可插拔 Loop 路由（A5） =====
        # agent→react（原生FC+降级）；code_act→代码即行动；deep_solve→计划先行+人工确认
        if req.mode in ("agent", "code_act", "deep_solve", "math"):
            from backend.agents.loops import get_loop, route_loop
            from backend.agents.loops.base import estimate_tokens
            system_context = f"{persona_prompt}\n\n{evolution_profile}\n\n{memory_prompt}\n\n{mode_prompt}"
            agent_sources: list = []

            # C1 上下文组成透明化（对标 dsh-context）：让学生看见 AI 是怎么理解自己的
            yield f"data: {json.dumps({'type': 'context_breakdown', 'persona': estimate_tokens(persona_prompt), 'profile': estimate_tokens(evolution_profile), 'memory': estimate_tokens(memory_prompt), 'mode': estimate_tokens(mode_prompt), 'history': estimate_tokens(''.join(h.get('content', '') for h in history)), 'total': estimate_tokens(system_context)}, ensure_ascii=False)}\n\n"

            loop = get_loop(route_loop(req.mode))
            awaiting_plan = False
            final_content = ""
            final_tokens = 0
            final_prompt_tokens = 0
            final_model = None

            async for agent_event in loop.run(
                message=req.message,
                history=history,
                system_context=system_context,
                user_id=current_user.id,
                conversation_id=conversation.id,
                plan_confirm=req.plan_confirm,
            ):
                event_type = agent_event.get("type", "")
                if event_type == "answer":
                    final_content = agent_event.get("content", "")
                    final_tokens = agent_event.get("tokens_used", 0)
                    final_prompt_tokens = agent_event.get("prompt_tokens", 0)
                    final_model = agent_event.get("model_used")
                elif event_type == "plan":
                    # 计划已生成，等待用户批准/修改/取消：本轮不落 AI 消息
                    awaiting_plan = True
                if event_type == "observation":
                    tool_data = agent_event.pop("data", None)
                    if agent_event.get("tool") == "knowledge_search" and agent_event.get("ok"):
                        seen = {s.get("chunk_id") for s in agent_sources}
                        for hit in (tool_data or {}).get("results", []):
                            if hit.get("chunk_id") not in seen:
                                agent_sources.append(hit)
                                seen.add(hit.get("chunk_id"))
                yield f"data: {json.dumps(agent_event, ensure_ascii=False)}\n\n"

            if awaiting_plan and not final_content:
                # 计划等待确认：不写 AI 消息，done 带 awaiting_plan 状态
                conversation.updated_at = datetime.utcnow()
                await db.commit()
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation.id, 'awaiting_plan': True}, ensure_ascii=False)}\n\n"
                return

            raw_reply = final_content
            final_content = content_safety_filter.filter_output(final_content)

            # 保存AI回复
            ai_msg = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=final_content,
                sources=agent_sources if agent_sources else None,
                latency_ms=int((time.time() - start_time) * 1000),
                model_used=final_model,
                tokens_used=final_tokens,
            )
            db.add(ai_msg)
            await _attach_review_flag(db, current_user.id, raw_reply, ai_msg)
            conversation.updated_at = datetime.utcnow()
            await db.commit()
            yield f"data: {json.dumps({'type': 'complete', 'full_content': final_content, 'sources': agent_sources, 'verification': {}, 'tokens_used': final_tokens, 'prompt_tokens': final_prompt_tokens, 'model_used': final_model}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation.id}, ensure_ascii=False)}\n\n"
            return

        # ===== 普通 chat 模式：直接流式 LLM 调用（实时思考过程 + 逐字回答） =====
        # 绕过 LangGraph 的非流式 normal_chat 节点，用 llm_client.stream() 原生 SSE 逐 token 输出
        # reasoning_delta 实时推送为 thinking 事件，content_delta 实时推送为 answer 事件
        _direct_stream = False
        if req.mode == "chat" and not req.skill_id and not req.knowledge_base_id:
            _direct_stream = True
            # 发送 session 事件，让前端拿到 conversation_id（否则第二条消息会变成新对话）
            yield f"data: {json.dumps({'type': 'session', 'conversation_id': conversation.id}, ensure_ascii=False)}\n\n"
            from backend.agents.llm_client import llm_client
            from datetime import datetime as _dt
            _current_time = _dt.now().strftime("%Y年%m月%d日 %H:%M:%S")
            _context_prompt = f"""当前系统时间：{_current_time}

请基于以下知识上下文回答学生的问题。如果上下文中没有相关信息，请明确说明。

知识上下文：
{knowledge_context[:2000]}

学生问题：{req.message}

要求：
1. 回答要准确、清晰
2. 如果引用了上下文，请标注来源
3. 不确定的地方要说明
"""
            if profile_prompt:
                _context_prompt = profile_prompt + "\n\n" + _context_prompt
            if mode_prompt:
                _context_prompt = mode_prompt + "\n\n" + _context_prompt
            if memory_prompt:
                _context_prompt = memory_prompt + "\n\n" + _context_prompt
            if evolution_profile:
                _evo_text = f"【学习者画像，仅供教学参考；画像中的词不是学生姓名，称呼学生一律用「你」】\n{evolution_profile}"
                if evolution_style:
                    _evo_text += f"\n学习风格: {evolution_style}"
                if evolution_level:
                    _evo_text += f"\n知识水平: {evolution_level}"
                _context_prompt = _evo_text + "\n\n" + _context_prompt
            if persona_prompt:
                _context_prompt = persona_prompt + "\n\n" + _context_prompt
            _messages = []
            for h in history[-5:]:
                _messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            _messages.append({"role": "user", "content": _context_prompt})

            _full_reasoning = ""
            _acc_content = ""
            final_content = ""
            final_tokens = 0
            final_prompt_tokens = 0
            final_model = None
            try:
                async for _chunk in llm_client.stream(_messages, deep_thinking=req.deep_thinking):
                    if _chunk.get("reasoning_delta"):
                        _full_reasoning += _chunk["reasoning_delta"]
                        yield f"data: {json.dumps({'type': 'thinking', 'content': _full_reasoning, 'is_real_reasoning': True, 'streaming': True}, ensure_ascii=False)}\n\n"
                    if _chunk.get("delta"):
                        _acc_content += _chunk["delta"]
                        yield f"data: {json.dumps({'type': 'answer', 'content': _acc_content}, ensure_ascii=False)}\n\n"
                    if _chunk.get("done"):
                        final_content = _chunk.get("content", _acc_content)
                        final_tokens = _chunk.get("tokens", 0)
                        final_prompt_tokens = _chunk.get("prompt_tokens", 0)
                        final_model = _chunk.get("model")
            except Exception as _e:
                logger.error(f"直接流式LLM调用异常: {_e}", exc_info=True)
                if not _acc_content:
                    _acc_content = f"抱歉，AI服务暂时不可用：{_e}"
                final_content = _acc_content
            final_reasoning = _full_reasoning
            final_sources = knowledge_sources
            yield f"data: {json.dumps({'type': 'complete', 'full_content': final_content, 'reasoning_content': final_reasoning, 'sources': final_sources, 'verification': {}, 'tokens_used': final_tokens, 'prompt_tokens': final_prompt_tokens, 'model_used': final_model}, ensure_ascii=False)}\n\n"

        # 流式运行LangGraph（普通模式，非直接流式时走这里）
        if not _direct_stream:
            async for event in run_tutor_graph_stream(
                user_id=current_user.id,
                conversation_id=conversation.id,
                message=req.message,
                mode=req.mode,
                history=history,
                knowledge_context=knowledge_context,
                deep_thinking=req.deep_thinking,
                extra_state={
                    "profile_prompt": profile_prompt,
                    "persona_id": persona_id,
                    "persona_prompt": persona_prompt,
                    "persona_behavior": persona_behavior,
                    "evolution_profile": evolution_profile,
                    "evolution_style": evolution_style,
                    "evolution_level": evolution_level,
                    "memory_prompt": memory_prompt,
                    "mode_prompt": mode_prompt,
                    "sources": knowledge_sources,
                    **learning_evidence,
                },
            ):
                if event["type"] == "complete":
                    final_content = event.get("full_content", "")
                    final_sources = event.get("sources", [])
                    final_verification = event.get("verification", {})
                    final_tokens = event.get("tokens_used", 0)
                    final_prompt_tokens = event.get("prompt_tokens", 0)
                    final_model = event.get("model_used") or None
                    bkt_pending = event.pop("bkt_update", None)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        raw_reply = final_content
        final_content = content_safety_filter.filter_output(final_content)
        if _qa_key and not _qa_cached and final_content and not _qa_reply_is_error(final_content):
            _qa_cache_put(_qa_key, {
                "reply": final_content,
                "sources": final_sources or [],
                "verification": final_verification or {},
                "model_used": final_model,
            })

        # 保存AI回复
        ai_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=final_content,
            sources=final_sources if final_sources else None,
            verification=final_verification if final_verification else None,
            latency_ms=int((time.time() - start_time) * 1000),
            model_used=final_model,
            tokens_used=final_tokens,
        )
        db.add(ai_msg)
        await _attach_review_flag(db, current_user.id, raw_reply, ai_msg)
        conversation.updated_at = datetime.utcnow()
        # 掌握度回灌：对话即改变BKT（与AI回复同一事务）
        mastery_update = await persist_graph_result(db, current_user.id, {"bkt_update": bkt_pending})
        await record_learning_session(
            db, current_user.id, conversation_id=conversation.id,
            mode=req.mode, started_ts=start_time, mastery_update=mastery_update,
        )
        await db.commit()

        # 掌握度回执：BKT 已落库，单独事件推送（complete 事件先于此，前端将回执补挂到末条消息）
        if mastery_update:
            kp_name = None
            try:
                kp = knowledge_graph.get_knowledge_point(str(mastery_update["knowledge_point_id"]))
                kp_name = (kp or {}).get("name")
            except Exception as e:
                logger.warning(f"知识点名称查询失败: {e}")
            yield f"data: {json.dumps({'type': 'mastery', 'mastery_update': {**mastery_update, 'kp_name': kp_name}}, ensure_ascii=False)}\n\n"

        # LLM 用量落库（api_key_usage：Key 消耗可追溯）
        if final_model:
            try:
                await record_api_usage(
                    db, current_user.id, final_model,
                    prompt_tokens=final_prompt_tokens or 0,
                    completion_tokens=max((final_tokens or 0) - (final_prompt_tokens or 0), 0),
                    endpoint="/api/chat/stream",
                )
            except Exception as e:
                logger.warning(f"API用量记录失败: {e}")

        # 学习闭环：检测是否需要加入错题本（优先用Verifier真实评估，降级关键词匹配）
        await maybe_add_wrong_question(db, current_user.id, conversation.id, req.message, final_content, final_verification)

        # 自适应难度调整（优先用Verifier真实评估，降级关键词匹配）
        await update_difficulty(db, profile, req.message, final_content, final_verification)
        # 上面的写入发生在主事务提交之后，必须再提交一次，否则close时回滚（错题本与难度从未落库）
        await db.commit()

        # ===== Hermes自进化 + Summary生成：fire-and-forget，done 事件不再等它们 =====
        # 此前这两步（各含一次 LLM 调用，read timeout 90s）串行在 done 之前，
        # 用户盯着最后一帧干等；后台化后对话响应立即收尾。
        # 注意：观察计数与融合触发逻辑保持原样，仅执行时机后移。
        _spawn_post_response_tasks(current_user.id, req.message, final_content, req.mode, final_verification)

        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation.id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def handle_command(cmd: dict, current_user: User, db: AsyncSession, req: ChatRequest) -> AsyncGenerator[str, None]:
    """处理斜杠命令"""
    command = cmd["command"]
    args = cmd["args"]

    if command == "test":
        # /test - 根据用户薄弱知识点动态生成测试题（LLM生成，失败回退模板）
        yield f"data: {json.dumps({'type': 'node_progress', 'node': 'quiz', 'node_name': '生成测试', 'description': '根据你的学习水平生成个性化测试题', 'status': 'running'}, ensure_ascii=False)}\n\n"
        try:
            from sqlalchemy import select as _select

            from backend.agents.llm_client import llm_client
            from backend.models import KnowledgeState
            # 查询用户掌握度最低的3个知识点
            weak_rows = (await db.execute(
                _select(KnowledgeState.knowledge_point_name, KnowledgeState.mastery_probability)
                .where(KnowledgeState.user_id == current_user.id)
                .order_by(KnowledgeState.mastery_probability.asc())
                .limit(3)
            )).all()
            weak_points = [f"{r[0]}（掌握度{r[1]:.0%}）" for r in weak_rows]
            topic = args or (weak_rows[0][0] if weak_rows else "当前学习内容")
            weak_desc = "、".join(weak_points) if weak_points else "暂无历史数据"
            quiz_prompt = (
                f"你是一位出题专家。学生当前薄弱知识点：{weak_desc}。"
                f"请围绕「{topic}」生成3道难度递进的练习题，"
                f"每道题包含：题目、难度标签（基础/中等/挑战）、详细答案解析。"
                f"用Markdown格式输出，题目编号清晰。"
            )
            resp = await llm_client.chat([{"role": "user", "content": quiz_prompt}], max_tokens=2048, use_cache=False)
            test_content = resp.get("content", "")
            if not test_content:
                raise ValueError("LLM返回为空")
        except Exception:
            # LLM不可用时回退模板
            test_content = (
                f"📝 **个性化测试题**\n\n基于你的学习配置，为你生成以下测试题（主题：{args or '当前知识点'}）：\n\n"
                "1. （选择题）以下哪个说法是正确的？\n   A. 选项A\n   B. 选项B\n   C. 选项C\n   D. 选项D\n\n"
                "2. （简答题）请解释核心概念的基本原理，并举例说明。\n\n"
                "3. （应用题）如何将所学知识应用到实际问题中？请给出具体步骤。\n\n"
                "💡 回答后我会批改并给出详细解析。\n\n*（注：AI生成服务暂不可用，显示模板题）*"
            )
        yield f"data: {json.dumps({'type': 'node_progress', 'node': 'quiz', 'node_name': '生成测试', 'status': 'completed'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'full_content': test_content, 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    elif command == "plan":
        # /plan - 创建学习计划（真实落库，否则"已生成"只是文案）
        from backend.models import LearningPlan
        from backend.plan.generator import plan_chapters

        topic = args or "新主题"
        profile = await get_user_profile(db, current_user.id)
        chapters, mode = await plan_chapters(
            db, current_user.id, title=topic,
            depth=getattr(getattr(profile, "learning_depth", None), "value", "high_school"),
        )
        plan = LearningPlan(
            user_id=current_user.id, title=topic,
            description=f"对话中创建：关于{topic}的学习计划",
            chapters=chapters, total_chapters=len(chapters),
            completed_chapters=0, progress_percent=0.0,
            current_chapter_id=chapters[0]["id"] if chapters else None,
        )
        db.add(plan)
        await db.commit()

        plan_content = f"📚 **学习计划已生成**（计划 #{plan.id}）\n\n主题：{topic}\n共{len(chapters)}个章节：\n\n"
        for ch in chapters:
            plan_content += f"⬜ **{ch['title']}**\n   {ch['description']}\n   知识点：{', '.join(ch['knowledge_points'])}\n\n"
        if mode == "template":
            plan_content += "⚠️ 本次 AI 章节规划未返回可用结构，已按结构化模板生成，可在「学习计划」页点「按学情重算」重新规划。\n\n"
        plan_content += "💡 在「学习计划」页面可以查看完整计划并跟踪进度。"
        yield f"data: {json.dumps({'type': 'complete', 'full_content': plan_content, 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    elif command == "answer":
        # /answer - 苏格拉底模式的"直接看答案"出口（分寸感三件套之一）
        # 学生主动看答案 ≠ 自主做对：命令路径不经 Verifier，天然不回灌 BKT
        from sqlalchemy import select as _select
        _rows = (await db.execute(
            _select(Message).where(Message.conversation_id == req.conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc()).limit(8)
        )).scalars().all()
        question = next((m.content for m in reversed(_rows) if m.role == "user"), "") if req.conversation_id else ""
        question = (args or question or "").strip()
        if not question:
            yield f"data: {json.dumps({'type': 'complete', 'full_content': '用法：`/answer 题目内容`，或在苏格拉底引导对话中直接点击「直接看答案」。', 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"
            return

        yield f"data: {json.dumps({'type': 'node_progress', 'node': 'tutor', 'node_name': '生成完整解答', 'description': '学生主动查看完整解答（本题不计入掌握度）', 'status': 'completed'}, ensure_ascii=False)}\n\n"
        from backend.agents.llm_client import llm_client
        try:
            resp = await llm_client.chat([
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": question[:2000]},
            ])
            answer_content = (resp.get("content") if isinstance(resp, dict) else str(resp)) or "抱歉，暂时无法生成解答，请稍后重试。"
        except Exception as e:
            logger.warning(f"/answer 生成失败: {e}")
            answer_content = "抱歉，暂时无法生成解答，请稍后重试。"
        answer_content = "📖 **完整解答**（你选择了直接查看答案，本题不计入掌握度统计）\n\n" + answer_content
        yield f"data: {json.dumps({'type': 'complete', 'full_content': answer_content, 'sources': [], 'verification': {}, 'answer_view': True}, ensure_ascii=False)}\n\n"

    elif command == "config":
        # /config - 显示/修改配置
        profile = await get_user_profile(db, current_user.id)
        config_content = "⚙️ **你的个性化配置**\n\n"
        config_content += f"- 学习深度：{profile.learning_depth.value}\n"
        config_content += f"- 学习风格：{profile.learning_style.value}\n"
        config_content += f"- 沟通方式：{profile.communication_style.value}\n"
        config_content += f"- 语气风格：{profile.tone_style.value}\n"
        config_content += f"- 推理框架：{profile.reasoning_framework.value}\n"
        config_content += f"- 当前难度：{profile.current_difficulty:.2f}\n"
        config_content += f"- 正确率：{profile.correct_questions}/{profile.total_questions}\n\n"
        config_content += "💡 在「个性化配置」页面可以修改这些设置。"
        yield f"data: {json.dumps({'type': 'complete', 'full_content': config_content, 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    elif command == "start":
        yield f"data: {json.dumps({'type': 'complete', 'full_content': '🚀 好的，我们开始学习！请告诉我你想学习什么主题，或者从学习计划中选择一个章节开始。', 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    elif command == "continue":
        yield f"data: {json.dumps({'type': 'complete', 'full_content': '✅ 好的，我们继续。请告诉我你上次学到哪里了，或者直接提问。', 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    elif command == "help":
        help_content = """📖 **可用命令**

`/test [知识点]` - 生成个性化测试题
`/plan [主题]` - 创建学习计划
`/config` - 查看个性化配置
`/start` - 开始学习
`/continue` - 继续学习
`/help` - 显示帮助

💡 也可以直接输入问题，AI会自动判断最合适的教学方式。"""
        yield f"data: {json.dumps({'type': 'complete', 'full_content': help_content, 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    else:
        yield f"data: {json.dumps({'type': 'complete', 'full_content': f'❓ 未知命令 /{command}。输入 /help 查看可用命令。', 'sources': [], 'verification': {}}, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


async def maybe_add_wrong_question(db: AsyncSession, user_id: int, conversation_id: str, question: str, answer: str, verification: dict | None = None):
    """学习闭环：检测对话中是否有错题，自动加入错题本并后台执行8维诊断
    优先使用LangGraph Verifier的真实LLM评估（incorrect+confidence>=0.7），
    无Verifier时降级为关键词启发式匹配。
    """
    is_wrong = False
    # 优先：Verifier真实评估
    if verification and verification.get("confidence", 0) >= 0.7 and verification.get("result") == "incorrect":
        is_wrong = True
    # 降级：关键词启发式（覆盖非LangGraph模式如research/agent/code_act等）
    elif not verification or not verification.get("result"):
        wrong_keywords = ["错误", "不对", "答错", "不正确", "错了", "失误"]
        if any(kw in answer for kw in wrong_keywords) and len(question) > 5:
            is_wrong = True

    if is_wrong and len(question) > 5:
        try:
            wrong = WrongQuestion(
                user_id=user_id,
                conversation_id=conversation_id,
                question=question[:500],
                student_answer="（对话中回答）",
                correct_answer=answer[:500],
                subject="通用",
                knowledge_point_name="待分析",
                diagnosis={"auto_detected": True, "source": "conversation", "status": "diagnosing"},
                diagnosis_detail="8维诊断执行中…",
            )
            db.add(wrong)
            await db.flush()
            wrong_id = wrong.id
            logger.info(f"自动加入错题本: user={user_id}, question={question[:30]}, id={wrong_id}, source={'verifier' if verification else 'keyword'}")
            # 后台异步执行真实8维LLM诊断（不阻塞对话响应）
            _spawn_wrong_diagnosis(user_id, wrong_id, question, answer)
        except Exception as e:
            logger.warning(f"自动加入错题本失败: {e}")


def _spawn_wrong_diagnosis(user_id: int, wrong_id: int, question: str, answer: str) -> None:
    """后台执行8维错题诊断并回写：fire-and-forget，失败只记日志"""
    task = asyncio.create_task(_run_wrong_diagnosis(user_id, wrong_id, question, answer))
    _POST_RESPONSE_TASKS.add(task)
    task.add_done_callback(_POST_RESPONSE_TASKS.discard)


async def _run_wrong_diagnosis(user_id: int, wrong_id: int, question: str, answer: str):
    """后台8维诊断：自持write_txn会话，调用LLM诊断器，回写诊断结果"""
    from backend.database import write_txn
    try:
        # 获取学生画像（认知水平+薄弱点）供诊断器使用
        async with write_txn() as bg_db:
            profile = await get_user_profile(bg_db, user_id)
            level = profile.knowledge_level if profile else "intermediate"
            from sqlalchemy import select
            from backend.models import KnowledgeState
            weak_rows = (await bg_db.execute(
                select(KnowledgeState.knowledge_point_name)
                .where(KnowledgeState.user_id == user_id)
                .order_by(KnowledgeState.mastery_probability.asc())
                .limit(3)
            )).all()
            weak_points = [r[0] for r in weak_rows]

        # 调用8维诊断器
        diagnosis = await wrong_diagnoser.diagnose(
            question=question,
            student_answer="（对话中回答）",
            reference_answer=answer,
            student_level=level,
            weak_points=weak_points,
        )

        # 回写诊断结果
        async with write_txn() as bg_db:
            wrong = await bg_db.get(WrongQuestion, wrong_id)
            if wrong:
                wrong.diagnosis = diagnosis.get("scores", {})
                wrong.primary_wrong_type = diagnosis.get("primary_type")
                wrong.diagnosis_detail = diagnosis.get("diagnosis_detail", "")
                if diagnosis.get("related_knowledge"):
                    wrong.knowledge_point_name = diagnosis["related_knowledge"][0][:200]
                await bg_db.commit()
                logger.info(f"8维错题诊断完成: wrong_id={wrong_id}, primary={diagnosis.get('primary_type')}")
    except Exception as e:
        logger.warning(f"8维错题诊断失败 wrong_id={wrong_id}: {e}")


async def update_difficulty(db: AsyncSession, profile: UserProfile, question: str, answer: str, verification: dict | None = None):
    """自适应难度：根据回答质量调整难度系数
    优先使用LangGraph Verifier的真实LLM评估（correct/incorrect+confidence>=0.7），
    无Verifier时降级为关键词启发式匹配。
    """
    if not profile.adaptive_difficulty_enabled:
        return

    profile.total_questions += 1

    is_correct = None  # True=答对, False=答错, None=不确定/不调整

    # 优先：Verifier真实评估
    if verification and verification.get("confidence", 0) >= 0.7:
        if verification.get("result") == "correct":
            is_correct = True
        elif verification.get("result") == "incorrect":
            is_correct = False
    # 降级：关键词启发式（覆盖非LangGraph模式）
    elif not verification or not verification.get("result"):
        correct_keywords = ["正确", "对了", "很棒", "完全正确", "非常好"]
        wrong_keywords = ["错误", "不对", "答错", "再想想", "不完全"]
        if any(kw in answer for kw in correct_keywords):
            is_correct = True
        elif any(kw in answer for kw in wrong_keywords):
            is_correct = False

    if is_correct is True:
        profile.correct_questions += 1
        # 答对了，适当提高难度（Verifier评估更可信，步长略大）
        step = 0.03 if verification else 0.02
        profile.current_difficulty = min(1.0, profile.current_difficulty + step)
    elif is_correct is False:
        # 答错了，适当降低难度
        step = 0.06 if verification else 0.05
        profile.current_difficulty = max(0.0, profile.current_difficulty - step)

    await db.flush()


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取用户的会话列表"""
    from sqlalchemy import desc, select
    result = await db.execute(
        select(Conversation).where(Conversation.user_id == current_user.id)
        .order_by(desc(Conversation.updated_at)).limit(50)
    )
    conversations = result.scalars().all()
    return [
        ConversationResponse(
            id=c.id,
            title=c.title or "新对话",
            mode=c.mode,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """获取会话消息（分页：默认最新50条，offset 向前翻历史）"""
    from sqlalchemy import select
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="会话不存在")

    # 总数（前端判断是否还有更早消息）
    from sqlalchemy import func
    total = (await db.execute(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    )).scalar() or 0

    # 取最新的 limit 条（按时间倒序取，再正序返回）
    result = await db.execute(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit).offset(offset)
    )
    messages = list(reversed(result.scalars().all()))
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(messages) < total,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "verification": m.verification,
                "tokens_used": m.tokens_used or 0,
                "model_used": m.model_used,
                "latency_ms": m.latency_ms,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_write_db),
):
    """删除会话"""
    from sqlalchemy import select
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")
    await db.delete(conversation)
    await db.commit()
    return None


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    req: RenameRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_write_db),
):
    """重命名会话（双击会话标题触发）"""
    from sqlalchemy import select
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")
    conversation.title = req.title.strip()
    await db.commit()
    return {"id": conversation.id, "title": conversation.title}


class FeedbackRequest(BaseModel):
    conversation_id: str | None = None
    message_index: int | None = None
    rating: str = Field(..., pattern="^(up|down)$")
    comment: str | None = None


@router.post("/feedback")
async def message_feedback(req: FeedbackRequest, current_user: User = Depends(get_current_user)):
    """消息级点赞/点踩反馈 → 写入进化画像观察，驱动自进化闭环"""
    obs = f"用户对AI回复点了{'赞' if req.rating == 'up' else '踩'}"
    if req.comment:
        obs += f"（备注：{req.comment[:100]}）"
    dialectic_profiler.add_observation(current_user.id, obs, source="feedback")
    return {"status": "recorded"}


class HintRequest(BaseModel):
    conversation_id: str
    level: int = Field(default=1, ge=1, le=3)


@router.post("/hint")
async def get_hint(
    req: HintRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """苏格拉底模式分级提示（ClueCard）：三档递进，始终不给最终答案

    分寸感设计：学生在引导中卡住时，自主选择求助档位——
    1=方向 / 2=步骤框架 / 3=关键公式。请求提示本身不推进会话阶段、不影响 BKT。
    """
    from sqlalchemy import select
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == req.conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = (await db.execute(
        select(Message).where(Message.conversation_id == req.conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc()).limit(6)
    )).scalars().all()
    question = next((m.content for m in reversed(rows) if m.role == "user"), "")
    if not question:
        raise HTTPException(status_code=400, detail="会话中没有可求助的问题")

    from backend.agents.llm_client import llm_client
    system = HINT_SYSTEM_PROMPT.format(level_rule=HINT_LEVEL_PROMPTS[req.level])
    try:
        resp = await llm_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"我的问题：{question[:2000]}"},
        ])
        hint = ((resp.get("content") if isinstance(resp, dict) else str(resp)) or "").strip()
    except Exception as e:
        logger.warning(f"提示生成失败: {e}")
        raise HTTPException(status_code=502, detail="提示生成失败，请稍后重试")
    return {"level": req.level, "hint": hint, "question": question[:200]}


# ===== Hermes自进化辅助函数 =====
# 后置学习闭环的后台任务注册表：强引用防 GC（任务被垃圾回收 = 静默丢失，
# 与资源中心 _BACKGROUND_JOBS 同一教训：后台任务必须被显式持有）
_POST_RESPONSE_TASKS: set = set()


def _spawn_post_response_tasks(user_id: int, message: str, ai_content: str, mode: str, verification: dict | None) -> None:
    """对话后的学习闭环（Hermes观察/融合 + Summary）后台化：
    1. 不阻塞用户拿到 done/响应；
    2. 自持 write_txn 短事务，不占用请求的写信号量坑位。
    """
    task = asyncio.create_task(_post_response_pipeline(user_id, message, ai_content, mode, verification))
    _POST_RESPONSE_TASKS.add(task)
    task.add_done_callback(_POST_RESPONSE_TASKS.discard)


async def _post_response_pipeline(user_id: int, message: str, ai_content: str, mode: str, verification: dict | None):
    """fire-and-forget 管道：失败只记日志，绝不影响对话主流程"""
    from backend.database import write_txn
    try:
        observation = f"用户提问: {message[:100]}。AI回复长度: {len(ai_content)}字。模式: {mode}。"
        if verification:
            observation += f"验证结果: {verification.get('result', 'unknown')}。"
        dialectic_profiler.add_observation(user_id, observation, source="dialog")

        evo_stats = dialectic_profiler.get_evolution_stats(user_id)
        if evo_stats["observation_count"] > 0 and evo_stats["observation_count"] % 3 == 0:
            logger.info(f"自动触发Hermes辩证融合: user={user_id}, observations={evo_stats['observation_count']}")
            # 融合需要写库（画像回写）：自开短事务会话，不碰请求的 db（请求会话此时可能已关闭）
            async with write_txn() as bg_db:
                await _trigger_synthesis(user_id, bg_db)
    except Exception as e:
        logger.warning(f"Hermes自进化观察失败: {e}")

    try:
        if ai_content and len(ai_content) > 20:
            await _generate_summary(user_id, message, ai_content, mode)
    except Exception as e:
        logger.warning(f"Summary生成失败: {e}")


async def _trigger_synthesis(user_id: int, db: AsyncSession):
    """辩证融合（与画像页"立即融合"共用同一服务，自动路径同样回写规范6维）"""
    try:
        result = await evolution_synthesis.run_synthesis(db, user_id)
        logger.info(
            f"Hermes辩证融合完成: user={user_id}, v{result['version']}, "
            f"conflict={result['conflict']}, 回写维度={result['applied'] or '无'}"
        )
    except evolution_synthesis.SynthesisError as e:
        logger.warning(f"Hermes辩证融合未完成: {e.detail}")
    except Exception as e:
        logger.warning(f"Hermes辩证融合失败: {e}")


async def _generate_summary(user_id: int, message: str, ai_response: str, mode: str):
    """异步生成学习进度摘要（不阻塞对话响应）"""
    try:
        from backend.agents.llm_client import llm_client
        prompt = summary_engine.build_summarize_prompt(user_id, message, ai_response, mode)
        response = await llm_client.chat([
            {"role": "system", "content": "你是一个学习进度分析专家。只输出JSON，不要任何解释。"},
            {"role": "user", "content": prompt},
        ])
        import json as _json
        import re as _re
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        json_match = _re.search(r'\{[\s\S]*\}', content)
        if json_match:
            summary = _json.loads(json_match.group())
            result = summary_engine.add_summary(user_id, summary)
            logger.info(f"Summary生成完成: user={user_id}, id={result.get('id')}, topics={len(summary.get('topics_learned', []))}")
    except Exception as e:
        logger.warning(f"Summary生成失败: {e}")


# ===== Agent 满血化新增端点（A4 审批 / A6 轨迹恢复 / B3 交互答题） =====

class ApprovalDecision(BaseModel):
    approved: bool


@router.post("/approvals/{approval_id}")
async def resolve_tool_approval(
    approval_id: str,
    body: ApprovalDecision,
    current_user: User = Depends(get_current_user),
):
    """A4 工具执行审批决定：SSE 流内 agent 等待此端点的用户决定后恢复执行"""
    from backend.agents.loops.base import ApprovalManager
    entry = ApprovalManager.get(approval_id)
    if not entry:
        raise HTTPException(status_code=404, detail="审批请求不存在或已处理")
    ok = ApprovalManager.resolve(approval_id, body.approved)
    return {"ok": ok, "approved": body.approved}


@router.get("/agent-runs")
async def get_agent_runs(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """A6 轨迹恢复：返回会话最近一次 agent 运行的完整轨迹（断线/刷新后前端重建面板）"""
    from backend.agents import checkpoints
    run = checkpoints.latest_run(conversation_id, current_user.id)
    if not run:
        return {"run": None}
    return {
        "run": {
            "id": run["id"],
            "mode": run["mode"],
            "loop": run["loop"],
            "status": run["status"],
            "plan": run.get("plan"),
            "steps": run["steps"],
            "created_at": run["created_at"],
        }
    }


class QuizAnswerRequest(BaseModel):
    conversation_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    selected: str = Field(..., max_length=2000)
    reference_answer: str = ""
    explanation: str = ""
    knowledge_point: str = ""


@router.post("/quiz-answer")
async def submit_quiz_answer(
    req: QuizAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_write_db),
):
    """B3 交互式答题闭环：回复内嵌选择题点击作答 → 判分 → 答错自动入错题本 + 记学习流水"""
    # 会话归属校验：conversation_id 必须属于当前用户（防跨用户写入错题/流水）
    from sqlalchemy import select as _select
    conv = (await db.execute(
        _select(Conversation.id).where(
            Conversation.id == req.conversation_id,
            Conversation.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    correct = bool(req.reference_answer) and req.selected.strip() == req.reference_answer.strip()

    if not correct:
        try:
            wrong = WrongQuestion(
                user_id=current_user.id,
                conversation_id=req.conversation_id,
                question=req.question[:500],
                student_answer=req.selected[:500],
                correct_answer=(req.reference_answer or "（见解析）")[:500],
                subject="通用",
                knowledge_point_name=req.knowledge_point[:100] or "待分析",
                diagnosis={"auto_detected": True, "source": "interactive_quiz"},
                diagnosis_detail=(req.explanation or "交互式答题自动收录")[:1000],
            )
            db.add(wrong)
            await db.flush()
            logger.info(f"交互答题错误入本: user={current_user.id}")
        except Exception as e:
            logger.warning(f"交互答题入本失败: {e}")

    # 学习流水：quiz 交互答题计入 streak/热图（复用复习评级同款链路）
    try:

        from backend.models import LearningSession
        db.add(LearningSession(
            user_id=current_user.id,
            conversation_id=req.conversation_id,
            mode="quiz",
            duration_seconds=0,
            questions_answered=1,
            correct_count=1 if correct else 0,
            ended_at=datetime.utcnow(),
        ))
    except Exception as e:
        logger.warning(f"quiz 流水记录失败: {e}")

    await db.commit()
    return {"correct": correct, "reference_answer": req.reference_answer,
            "explanation": req.explanation}
