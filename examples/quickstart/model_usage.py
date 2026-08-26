#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemBook Model Usage Example

展示如何使用不同的 LLM 模型（OpenAI API 或本地 VLLM）与 MemBook 框架。
"""

import os

from membook import (
    MemoryAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
)


def openai_api_example():
    """OpenAI API 模型使用示例"""
    print("=== OpenAI API 模型示例 ===\n")
    
    # 1. 配置 OpenAI Generator
    gen_config = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",  # 或 "gpt-4", "gpt-3.5-turbo"
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=1000
    )
    
    # 2. 创建 Generator
    generator = OpenAIGenerator.from_config(gen_config)
    
    # 3. 创建存储
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()
    
    # 4. 创建 MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )
    
    # 5. 测试简单文档
    documents = [
        "机器学习是人工智能的一个子集。",
        "深度学习使用多层神经网络。",
        "自然语言处理专注于人类语言理解。"
    ]
    
    print(f"正在处理 {len(documents)} 个文档...")
    for doc in documents:
        memory_agent.memorize(doc)
    
    memory_state = memory_store.load()
    print(f"✅ 构建了 {len(memory_state.abstracts)} 个记忆摘要\n")
    
    return True


def custom_api_endpoint_example():
    """自定义 API 端点示例（兼容 OpenAI 的第三方服务）"""
    print("=== 自定义 API 端点示例 ===\n")
    
    # 1. 配置自定义端点的 OpenAI Generator
    gen_config = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://your-custom-endpoint.com/v1",  # 自定义端点
        temperature=0.3
    )
    
    # 2. 创建 Generator
    generator = OpenAIGenerator.from_config(gen_config)
    
    # 3. 创建存储
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()
    
    # 4. 创建 MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )
    
    print("✅ 配置了自定义 API 端点")
    print(f"   端点: {gen_config.base_url}")
    print(f"   模型: {gen_config.model_name}\n")
    
    return True


def vllm_local_model_example():
    """VLLM 本地模型使用示例"""
    print("=== VLLM 本地模型示例 ===\n")
    
    try:
        from membook import VLLMGenerator, VLLMGeneratorConfig
        
        # 1. 配置 VLLM Generator
        gen_config = VLLMGeneratorConfig(
            model_name="Qwen2.5-7B-Instruct",  # 本地模型名称
            api_key="empty",  # VLLM 通常使用 "empty" 作为 API Key
            base_url="http://localhost:8000/v1",  # vLLM 服务器地址
            temperature=0.7,
            max_tokens=512
        )
        
        # 2. 创建 Generator
        generator = VLLMGenerator.from_config(gen_config)
        
        # 3. 创建存储
        memory_store = InMemoryMemoryStore()
        page_store = InMemoryPageStore()
        
        # 4. 创建 MemoryAgent
        memory_agent = MemoryAgent(
            generator=generator,
            memory_store=memory_store,
            page_store=page_store
        )
        
        # 5. 测试简单文档
        documents = [
            "AI is artificial intelligence.",
            "ML is machine learning.",
            "DL is deep learning."
        ]
        
        print(f"正在处理 {len(documents)} 个文档...")
        for doc in documents:
            memory_agent.memorize(doc)
        
        memory_state = memory_store.load()
        print(f"✅ 构建了 {len(memory_state.abstracts)} 个记忆摘要\n")
        
        return True
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: pip install vllm>=0.6.0")
        return False
    except Exception as e:
        print(f"❌ 本地模型错误: {e}")
        print("   提示: 如果内存有限，尝试使用更小的模型")
        return False


def model_comparison():
    """模型对比说明"""
    print("\n=== 模型选择指南 ===\n")
    
    print("📌 OpenAI API 模型:")
    print("   优点:")
    print("     ✅ 快速开始，无需本地资源")
    print("     ✅ 强大的性能和准确性")
    print("     ✅ 自动更新和维护")
    print("   缺点:")
    print("     ❌ 需要网络连接")
    print("     ❌ 按使用量付费")
    print("     ❌ 数据发送到外部服务器")
    print()
    
    print("📌 VLLM 本地模型:")
    print("   优点:")
    print("     ✅ 完全离线运行")
    print("     ✅ 数据隐私保护")
    print("     ✅ 无使用限制")
    print("   缺点:")
    print("     ❌ 需要 GPU 资源")
    print("     ❌ 需要下载和管理模型")
    print("     ❌ 可能需要更多配置")
    print()
    
    print("💡 建议:")
    print("   - 快速原型和开发: 使用 OpenAI API")
    print("   - 生产环境和隐私要求: 考虑本地 VLLM")
    print("   - 大规模使用: 根据成本和性能权衡选择")


def main():
    """主函数"""
    print("=" * 60)
    print("MemBook 模型使用示例")
    print("=" * 60)
    print()
    
    # 检查 API Key
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    if not has_api_key:
        print("⚠️  未检测到 OPENAI_API_KEY 环境变量")
        print("   某些示例将无法运行")
        print("   设置方法: export OPENAI_API_KEY='your-api-key'\n")
    
    # 测试 OpenAI API 模型
    if has_api_key:
        try:
            openai_success = openai_api_example()
        except Exception as e:
            print(f"OpenAI API 示例失败: {e}\n")
            openai_success = False
    else:
        print("跳过 OpenAI API 示例（未设置 API Key）\n")
        openai_success = False
    
    # 自定义端点示例（仅配置，不实际运行）
    custom_endpoint_success = custom_api_endpoint_example()
    
    # 测试 VLLM 本地模型（可选）
    print("是否测试 VLLM 本地模型？")
    test_vllm = input("输入 'yes' 继续，或按 Enter 跳过: ").strip().lower()
    
    if test_vllm == 'yes':
        vllm_success = vllm_local_model_example()
    else:
        print("跳过 VLLM 本地模型示例\n")
        vllm_success = False
    
    # 显示模型对比
    model_comparison()
    
    # 总结
    print("\n" + "=" * 60)
    print("示例总结")
    print("=" * 60)
    if openai_success:
        print("✅ OpenAI API 模型: 适合快速原型和部署")
    if custom_endpoint_success:
        print("✅ 自定义端点: 适合使用兼容 OpenAI 的第三方服务")
    if vllm_success:
        print("✅ VLLM 本地模型: 适合隐私和离线使用")
    
    print("\n💡 根据你的需求选择合适的模型类型！")
    print("\n更多信息:")
    print("  - 查看 eval/ 目录了解评估示例")
    print("  - 查看 gam/generator/ 了解生成器实现")


if __name__ == "__main__":
    main()
