# Gemini Prompt 资源整理说明

该目录用于汇总 `gemini prompt.md` 中需要替换/引用的图示素材。

## 文件清单

1. `gemini prompt.md`
   - 原始 Gemini 提示词文档（14 页答辩文案，含致谢页）。

2. `有效的一张ppt样例.png`
   - 已验证效果较好的单页样例图。
   - 用途：给 Gemini 作为风格参考图，帮助其学习信息层级、留白、结论横条、主图容器与底部 bullet 收口。
   - 注意：只参考视觉语言与排版逻辑，不直接复制原图构图。

3. `page3_训练推理架构图.png`
   - 第 3 页可直接使用的图片文件（PNG，受支持格式）。

4. `page10_线上增强架构图.png`
   - 第 10 页可直接使用的图片文件（PNG，受支持格式）。
   - 关键信息来源：
     - `src/inference/plan_b.py`
     - `src/inference/remote_explainer.py`
     - `distill/run_llm_distill_pipeline.py`
     - `distill/collect_llm_teacher.py`
     - `distill/build_cache_from_llm.py`
     - `train_multitask.py`

5. `线上大模型辅助增强方案.png`
   - 第 10 页推荐参考图。
   - 用途：让 Gemini 参考线上增强部分的结构组织方式，但页面最终只展示方案 A / B / C，不提及 Plan D，也不要直接照搬原图。

## 建议替换映射

- 风格参考图 → 使用 `有效的一张ppt样例.png`
- 第 3 页“训练/推理架构图” → 使用 `page3_训练推理架构图.png`
- 第 10 页“线上大模型辅助增强方案” → 优先参考 `线上大模型辅助增强方案.png`

## 备注

- 目录中原有 `训练推理架构图.html` 与 `page10_线上增强架构图.svg` 保留为源素材。
- 如果目标平台只支持你列出的格式，请优先上传两个 PNG 文件。
