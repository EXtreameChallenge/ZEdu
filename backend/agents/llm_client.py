"""
LLM客户端 — 多Key轮询 + 降级链 + 超时 + 缓存 + 重试
审核修正：确保API调用失败时有完整的降级策略
"""
import asyncio
import hashlib
import json
import time
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from backend.config import settings


def _retryable(exc) -> bool:
    """只重试瞬态故障：连接/读写超时与限流、网关类状态码；鉴权失败重试无意义
    注意：retry_if_exception() 传入的是异常对象本身，不是 retry_state"""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _is_placeholder_key(value: str) -> bool:
    """占位符 Key 识别：README/模板里遗留的示例值进了降级链只会白等一次 401

    历史教训：.env 模板默认 DASHSCOPE_API_KEY=your-dashscope-api-key，
    用户没填真值，降级链每次都拿它打阿里云接口拿 401——失败前白耗 0.1~1s，
    且日志里"降级也失败"会误导排查（看起来像通道坏了，其实是没配）。
    """
    if not value or not value.strip():
        return True
    v = value.strip().lower()
    placeholders = ("your-", "your_", "your ", "xxx", "changeme", "change-me", "placeholder", "replace", "<", ">", "todo", "sk-xxx")
    return any(p in v for p in placeholders)


class LLMClient:
    """
    统一LLM调用客户端
    - 多API Key轮询（避免单Key限流）
    - 降级链：当前用户所选模型 → glm-4-9b → 通义千问 → 星火
    - 响应缓存（相同问题直接返回缓存）
    - 超时控制（connect 10s / read 90s，长回答不会被 30s 掐断）
    - tenacity 指数退避重试（仅传输错误与 429/5xx）
    """

    def __init__(self):
        self.api_keys = settings.zhipu_key_list
        self.current_key_index = 0
        self.base_url = settings.zhipu_base_url
        self.model = settings.zhipu_model
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_max_size = 1000
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0))
        return self._client

    def _get_next_key(self) -> str | None:
        """轮询获取下一个API Key"""
        if not self.api_keys:
            return None
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def _cache_key(self, messages: list[dict], model: str, temperature: float,
                   tools: list[dict] | None = None) -> str:
        """生成缓存键（工具目录参与哈希：同一问题不同工具集结果不同）"""
        content = str(messages) + model + str(temperature)
        if tools:
            content += hashlib.md5(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _global_settings_model() -> dict[str, Any]:
        """本次调用的模型配置：优先请求入口绑定的每用户覆盖，其次全局默认值"""
        try:
            from backend.settings.model_ctx import get_model_override
            override = get_model_override()
            if override:
                return override
        except Exception:
            pass
        try:
            from backend.settings.routes import load_settings
            return load_settings().get("model") or {}
        except Exception:
            return {}

    @staticmethod
    def _model_supports_thinking(model: str) -> bool:
        """判断模型是否原生支持 thinking 推理链（GLM-4.5 及以上系列）"""
        if not model:
            return False
        m = model.lower()
        return any(k in m for k in ['glm-4.5', 'glm-4-5', 'glm-4-plus', 'glm-4.6v', 'glm-4.1v-thinking'])

    def _resolve_thinking_model(self, target_model: str, deep_thinking: bool) -> str:
        """深度思考自动切换：开启思考但当前模型不支持时，切到 thinking_model

        仿 DeepSeek 设计：用户选了快速模型（如 glm-4-flash），开启深度思考后
        自动切到支持推理链的模型（glm-4.5-air），关闭思考时保持原模型。
        """
        if not deep_thinking:
            return target_model
        if self._model_supports_thinking(target_model):
            return target_model
        thinking_model = getattr(settings, 'zhipu_thinking_model', 'glm-4.5-air')
        if thinking_model and thinking_model != target_model:
            logger.info(f"深度思考已开启，模型从 {target_model} 自动切换到 {thinking_model}")
        return thinking_model or target_model

    @staticmethod
    def _usage_tokens(data: dict[str, Any]) -> dict[str, int]:
        """归一各通道usage：OpenAI兼容协议用prompt_tokens，通义用input_tokens"""
        usage = data.get("usage") or {}
        return {
            "tokens": usage.get("total_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
        tools: list[dict[str, Any]] | None = None,
        deep_thinking: bool = True,
    ) -> dict[str, Any]:
        """统一聊天接口（原生 function calling 支持，A1）"""
        cfg = self._global_settings_model()
        if model is None:
            model = cfg.get("model_name")
        if temperature is None:
            temperature = cfg.get("temperature", 0.7)
        if max_tokens is None:
            max_tokens = cfg.get("max_tokens", 4096)
        base_model = model or self.model
        target_model = base_model
        provider = cfg.get("provider") or "zhipu"

        # 深度思考自动切换模型（仿 DeepSeek：开思考时自动用支持推理链的模型）
        target_model = self._resolve_thinking_model(target_model, deep_thinking)

        # 当前用户选择星火/通义通道且服务端已配置真实Key时，走对应接口；失败回落智谱
        if provider == "xinghuo" and settings.xinghuo_api_key and not _is_placeholder_key(settings.xinghuo_api_key):
            try:
                fb = await self._call_xinghuo(messages, target_model, temperature, max_tokens, tools=tools)
                fb["cached"] = False
                return fb
            except Exception as e:
                logger.warning(f"Xinghuo channel failed, fallback to zhipu: {e}")
        if provider == "dashscope" and settings.dashscope_api_key and not _is_placeholder_key(settings.dashscope_api_key):
            try:
                fb = await self._call_dashscope(messages, target_model, temperature, max_tokens)
                fb["cached"] = False
                return fb
            except Exception as e:
                logger.warning(f"Dashscope channel failed, fallback to zhipu: {e}")
        cache_key = self._cache_key(messages, target_model, temperature, tools)

        # 1. 缓存命中
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"LLM cache hit: {cache_key[:8]}")
            return {**cached, "cached": True}

        # 2. 尝试主模型（多Key轮询）
        try:
            result = await self._call_zhipu(messages, target_model, temperature, max_tokens, tools=tools, deep_thinking=deep_thinking)
            result["cached"] = False
            if use_cache and len(self._cache) < self._cache_max_size:
                self._cache[cache_key] = {k: v for k, v in result.items() if k != "cached"}
            return result
        except Exception as e:
            logger.warning(f"主模型调用失败: {e}，尝试降级")

        # 3. 降级链：异构通道优先（H4 缓解）
        # 深度思考自动切换模型失败时（如 glm-4.5-air 余额不足），先回退原模型重试一次
        fallback_models = []
        if base_model and base_model != target_model:
            fallback_models.append((base_model, self._call_zhipu))
        if settings.dashscope_api_key and settings.dashscope_model and not _is_placeholder_key(settings.dashscope_api_key):
            fallback_models.append((settings.dashscope_model, self._call_dashscope))
        if settings.xinghuo_api_key and settings.xinghuo_model and not _is_placeholder_key(settings.xinghuo_api_key):
            fallback_models.append((settings.xinghuo_model, self._call_xinghuo))
        fallback_models.append(("glm-4-9b", self._call_zhipu))
        if len(fallback_models) == 1:
            logger.warning("降级链仅剩同源 glm-4-9b：未配置通义/星火 Key，智谱故障时无真正异构备份")
        for fb_model, fb_func in fallback_models:
            if not fb_model:
                continue
            try:
                logger.info(f"降级到模型: {fb_model}")
                result = await fb_func(messages, fb_model, temperature, max_tokens)
                result["cached"] = False
                result["fallback_used"] = True
                return result
            except Exception as e:
                logger.warning(f"降级模型 {fb_model} 也失败: {e}")
                continue

        # 4. 全部失败
        logger.error("所有LLM模型调用失败")
        if settings.demo_mode:
            logger.info("演示模式：返回模拟回复")
            mock_content = self._generate_mock_reply(messages)
            return {
                "content": mock_content,
                "model": "demo-mock",
                "tokens": len(mock_content),
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
                "cached": False,
                "demo": True,
            }

        return {
            "content": "抱歉，AI服务暂时不可用，请稍后重试。",
            "model": "error",
            "tokens": 0,
            "prompt_tokens": 0,
            "cached": False,
            "error": True,
        }

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deep_thinking: bool = True,
    ):
        """Token 级流式输出（async generator）。"""
        cfg = self._global_settings_model()
        if model is None:
            model = cfg.get("model_name")
        if temperature is None:
            temperature = cfg.get("temperature", 0.7)
        if max_tokens is None:
            max_tokens = cfg.get("max_tokens", 4096)
        base_model = model or self.model
        target_model = base_model
        provider = cfg.get("provider") or "zhipu"

        # 深度思考自动切换模型（仿 DeepSeek：开思考时自动用支持推理链的模型）
        target_model = self._resolve_thinking_model(target_model, deep_thinking)

        # 1. 异构通道（星火/通义）：无原生流式，降级为一次性输出
        if provider == "xinghuo" and settings.xinghuo_api_key and not _is_placeholder_key(settings.xinghuo_api_key):
            try:
                result = await self._call_xinghuo(messages, target_model, temperature, max_tokens)
                async for chunk in self._yield_full(result):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Xinghuo stream failed, fallback: {e}")
        if provider == "dashscope" and settings.dashscope_api_key and not _is_placeholder_key(settings.dashscope_api_key):
            try:
                result = await self._call_dashscope(messages, target_model, temperature, max_tokens)
                async for chunk in self._yield_full(result):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Dashscope stream failed, fallback: {e}")

        # 2. 智谱原生 SSE 流式
        try:
            async for chunk in self._stream_zhipu(messages, target_model, temperature, max_tokens, deep_thinking=deep_thinking):
                yield chunk
            return
        except Exception as e:
            logger.warning(f"智谱流式调用失败: {e}，尝试降级")

        # 3. 降级链（非流式，一次性输出）：thinking 切换失败先回退原模型，占位符 Key 跳过
        fallback_models = []
        if base_model and base_model != target_model:
            fallback_models.append((base_model, self._call_zhipu))
        if settings.dashscope_api_key and settings.dashscope_model and not _is_placeholder_key(settings.dashscope_api_key):
            fallback_models.append((settings.dashscope_model, self._call_dashscope))
        if settings.xinghuo_api_key and settings.xinghuo_model and not _is_placeholder_key(settings.xinghuo_api_key):
            fallback_models.append((settings.xinghuo_model, self._call_xinghuo))
        fallback_models.append(("glm-4-9b", self._call_zhipu))
        for fb_model, fb_func in fallback_models:
            if not fb_model:
                continue
            try:
                logger.info(f"流式降级到: {fb_model}")
                result = await fb_func(messages, fb_model, temperature, max_tokens)
                result["fallback_used"] = True
                async for chunk in self._yield_full(result):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"降级模型 {fb_model} 也失败: {e}")
                continue

        # 4. 全部失败
        if settings.demo_mode:
            mock = self._generate_mock_reply(messages)
            async for chunk in self._yield_full({"content": mock, "model": "demo-mock", "tokens": len(mock), "prompt_tokens": 0, "demo": True}):
                yield chunk
            return
        async for chunk in self._yield_full({"content": "抱歉，AI服务暂时不可用，请稍后重试。", "model": "error", "tokens": 0, "prompt_tokens": 0, "error": True}):
            yield chunk

    @staticmethod
    async def _yield_full(result: dict[str, Any]):
        """非流式降级：把完整回复拆成若干小片段 yield，模拟打字机效果"""
        content = result.get("content", "")
        reasoning = result.get("reasoning_content", "")
        if reasoning:
            yield {"delta": "", "reasoning_delta": reasoning, "done": False}
        chunk_size = 8
        for i in range(0, len(content), chunk_size):
            yield {"delta": content[i:i + chunk_size], "done": False}
            await asyncio.sleep(0.01)
        yield {
            "delta": "", "done": True,
            "content": content,
            "model": result.get("model", ""),
            "tokens": result.get("tokens", 0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "fallback_used": result.get("fallback_used", False),
            "demo": result.get("demo", False),
            "error": result.get("error", False),
        }

    async def _stream_zhipu(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int,
        deep_thinking: bool = True,
    ):
        """智谱 AI 原生 SSE 流式调用，逐 token yield"""
        api_key = self._get_next_key()
        if not api_key:
            raise ValueError("没有可用的智谱API Key")

        client = await self._get_client()
        start_time = time.time()
        full_content = ""
        full_reasoning = ""
        usage = {"tokens": 0, "prompt_tokens": 0}

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if deep_thinking and self._model_supports_thinking(model):
            payload["thinking"] = {"type": "enabled"}

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content_delta = delta.get("content", "")
                reasoning_delta = delta.get("reasoning_content", "")
                if reasoning_delta:
                    full_reasoning += reasoning_delta
                    yield {"delta": "", "reasoning_delta": reasoning_delta, "done": False}
                if content_delta:
                    full_content += content_delta
                    yield {"delta": content_delta, "done": False}
                if data.get("usage"):
                    usage = self._usage_tokens(data)

        latency_ms = int((time.time() - start_time) * 1000)
        yield {
            "delta": "", "done": True,
            "content": full_content,
            "reasoning_content": full_reasoning,
            "model": model,
            "tokens": usage["tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "latency_ms": latency_ms,
        }

    def _generate_mock_reply(self, messages: list[dict[str, str]]) -> str:
        """生成模拟回复（演示模式用）"""
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        if not user_msg:
            return "你好！我是ZEdu Duo AI助教。有什么学习问题可以问我。"

        msg_lower = user_msg.lower()
        if any(k in msg_lower for k in ["你好", "hi", "hello", "在吗"]):
            return "你好！我是ZEdu Duo AI助教，很高兴为你服务。你可以问我任何学习问题，我会通过多智能体团队为你提供个性化辅导。"
        if any(k in msg_lower for k in ["数学", "math", "函数", "方程", "几何"]):
            return f"关于「{user_msg}」这个数学问题，我来为你分析：\n\n**解题思路：**\n1. 首先明确题目给出的已知条件\n2. 确定需要求解的目标量\n3. 选择合适的数学方法（公式/定理）\n4. 逐步推导并验证结果\n\n**建议：** 多做同类题目，总结解题套路。如果需要更详细的讲解，可以告诉我具体的题目内容。\n\n> 这是演示模式的模拟回复，配置API Key后可获得真实AI解答。"
        if any(k in msg_lower for k in ["英语", "english", "单词", "语法", "翻译"]):
            return f"关于「{user_msg}」这个英语问题：\n\n**学习建议：**\n- 词汇：结合语境记忆，不要死记硬背\n- 语法：理解规则后多造句练习\n- 阅读：每天坚持读一篇英文文章\n\n> 这是演示模式的模拟回复，配置API Key后可获得真实AI解答。"
        if any(k in msg_lower for k in ["物理", "physics", "力", "电", "运动"]):
            return f"关于「{user_msg}」这个物理问题：\n\n**分析步骤：**\n1. 确定研究对象和物理过程\n2. 受力分析/运动分析\n3. 选择合适的物理定律\n4. 列方程求解\n\n> 这是演示模式的模拟回复，配置API Key后可获得真实AI解答。"

        return f"收到你的问题：「{user_msg}」\n\n我是ZEdu Duo的多智能体AI助教团队。在演示模式下，我为你提供基础的学习指导：\n\n1. **明确问题**：把大问题拆成小问题\n2. **查找资料**：查阅相关知识点\n3. **尝试解答**：先自己思考，再对照答案\n4. **总结反思**：记录错题和解题方法\n\n如果你需要更深入的辅导，请配置智谱AI API Key（在backend/.env中填写ZHIPU_API_KEYS），即可获得真实的多智能体AI辅导体验。"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception(_retryable),
        reraise=True,
    )
    async def _call_zhipu(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        deep_thinking: bool = True,
    ) -> dict[str, Any]:
        """调用智谱AI API（OpenAI 兼容；tools 原生 function calling）"""
        api_key = self._get_next_key()
        if not api_key:
            raise ValueError("没有可用的智谱API Key")

        client = await self._get_client()
        start_time = time.time()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if deep_thinking and self._model_supports_thinking(model):
            payload["thinking"] = {"type": "enabled"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = int((time.time() - start_time) * 1000)
        choice = data["choices"][0]
        message = choice.get("message", {})

        return {
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "model": data.get("model", model),
            **self._usage_tokens(data),
            "latency_ms": latency_ms,
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
        }

    async def _call_dashscope(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        """调用通义千问API（降级用）"""
        if not settings.dashscope_api_key:
            raise ValueError("没有配置通义千问API Key")

        client = await self._get_client()
        start_time = time.time()

        response = await client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers={
                "Authorization": f"Bearer {settings.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": {"messages": messages},
                "parameters": {"temperature": temperature, "max_tokens": max_tokens},
            },
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "content": data["output"]["choices"][0]["message"]["content"],
            "model": model,
            **self._usage_tokens(data),
            "latency_ms": latency_ms,
        }

    async def _call_xinghuo(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """调用讯飞星火API（V2.1 OpenAI兼容端点，支持 tools）"""
        if not settings.xinghuo_api_key:
            raise ValueError("没有配置讯飞星火API Key")

        client = await self._get_client()
        start_time = time.time()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = await client.post(
            f"{settings.xinghuo_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.xinghuo_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = int((time.time() - start_time) * 1000)
        choice = data["choices"][0]
        message = choice.get("message", {})
        return {
            "content": message.get("content") or "",
            "model": model,
            **self._usage_tokens(data),
            "latency_ms": latency_ms,
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# 全局单例
llm_client = LLMClient()
