目标：根据项目文档和代码信息制作汇报 html，用类似于 ppt 的分页组织内容，风格为暖白色精简风格，

注意：以下仅仅是大纲，具体内容（文字、图片等）需要你自己根据当前项目里各个文档以及代码等一切可以利用的信息进行自由的组织编排，但请注意不要编造。允许直接采用现成的图片等。

以下是每页内容大纲：
1. 封面：多媒体技术课设答辩，答辩人：谢鹏、陈冠宇
2. 数据集选择：原本打算直接采用课程参考数据集，但是它们都缺少分类信息，因此上网寻找包含分类信息的数据集，发现 GenImage 恰好拥有分类信息但是缺少解释和定位信息。两者恰好互补，因此我们决定混合使用两个数据集进行训练。
3. 训练架构：因为是混合数据集，我们设计了这样一个架构（展示设计架构，请你贴上图片或者自己生成）
4. 训练策略：分 A 和 B 两阶段“先热身再联合”，每次修改后先跑冒烟测试确保链路畅通，然后在进行正式训练
5. 第一次冒烟测试：<结果>，说明当前 smoke test 已经跑通分类、解释、定位三条链路，其中分类效果很好，解释任务已有初步相似度结果，但定位任务目前结果为 0，需要后续重点排查。用冒烟测试的训练结果测试验收数据集：<结果> 还是有问题。
6. 问题修复：通过排查修复了问题<原因> <修复方式> ，重新跑了smoke训练与评估，定位指标恢复非零。修复后再次单图推理，预测 mask 不再是纯黑图，说明定位监督链路已经恢复正常
7. 诊断分类指标偏低问题：在跑了第一次正式测试后发现分类指标偏低（测试结果没有备份）。诊断课程数据集上真假分类偏低的问题，统计发现课程 fake 图像来源不只一种，而是包含 `adm`、`biggan`、`glide/vqdm(1000)`、`midjourney`、`sdv4`、`sdv5`、`wukong` 等多类生成器。进一步统计训练用的 `data/imagenet_ai_0508_adm` 后确认：当前分类训练 fake 数据实际上只有 `adm` 一种生成器，`train/ai` 为 162000 张 `adm`，`val/ai` 为 6000 张 `adm`。由此确认课程分类表现偏低的关键原因之一是生成器域偏移，而不只是阈值问题。新增接入 `data/imagenet_ai_0419_biggan` 作为第二个分类训练数据源，并把配置项从单个 `imagenet_root` 扩展为多个 `imagenet_roots`（这部分内容有点多，自行精简取舍）
8. 第二次冒烟测试：重新运行多数据源配置下的 smoke 训练，确认训练流程仍然可以正常跑通，最终得到：
  - Acc = 0.522350013256073
  - AP = 0.4470061957836151
  - IoU = 0.07516614347696304
  - PixP = 0.15188011527061462
  - PixR = 0.18840335309505463
  - PixF1 = 0.16818168171894715
  - BLEU = 0.426003784020339
  - ROUGE-L = 0.32699134048865086
说明当前多数据源分类训练改动已经接线成功；由于 smoke 配置每类只取极少样本，当前 smoke 的分类指标主要用于验证流程，不代表正式训练上限（内容较多，自行取舍）
9. 第二次正式测试：重新完成基于修复后监督和多数据源分类配置的正式训练，并用新的 `outputs/best.pt` 在课程验收数据集上重新评估，得到：
  - Acc = 0.699999988079071
  - AP = 0.8362857699394226
  - BLEU = 0.4270665618998833
  - ROUGE-L = 0.2985437231203506
  - IoU = 0.150065615773201
  - PixP = 0.34466275572776794
  - PixR = 0.267295777797699
  - PixF1 = 0.30108869906401314
- 对新的课程验收结果做了解读，确认相比之前课程评估结果，分类与定位都有明显提升，解释指标整体保持稳定，说明多数据源分类训练与定位监督修复都产生了正向效果
- 进一步按课程分类数据中的生成器类型拆分了模型表现，统计结果如下：
  - adm：Acc = 0.88，fake recall = 0.88，avg fake score = 0.8766
  - biggan：Acc = 1.00，fake recall = 1.00，avg fake score = 1.0000
  - glide/vqdm1000：Acc = 0.12，fake recall = 0.12，avg fake score = 0.1595
  - midjourney：Acc = 0.16，fake recall = 0.16，avg fake score = 0.1886
  - sdv4：Acc = 0.64，fake recall = 0.64，avg fake score = 0.5880
  - sdv5：Acc = 0.40，fake recall = 0.40，avg fake score = 0.4350
  - wukong：Acc = 0.24，fake recall = 0.24，avg fake score = 0.2800
  - real：Acc = 0.955，avg fake score = 0.0731
由生成器拆分结果进一步确认：当前分类主问题不是阈值，而是生成器覆盖不足。模型已经学会 `adm` 和 `biggan`，但对 `glide/vqdm1000`、`midjourney`、`wukong` 等未覆盖生成器几乎失效，后续应优先补这些 fake 数据源，而不是先折腾阈值
10. 线上模型辅助增强：计划利用线上模型增强效果，设计了4个方案（参看<训练推理架构图.html>），最后尝试了前三个方案
11. 方案 A （解释增强）：本地模型继续负责真假分类和定位，线上多模态大模型只根据 原图 + 预测 mask + fake_score + 检索到的 top-k caption 生成更自然、更细致的中文解释文本。效果：- 全量 200 张 explanation 对比结果如下：
  - local BLEU = 0.4647
  - remote BLEU = 0.3806
  - local ROUGE-L = 0.3016
  - remote ROUGE-L = 0.2986
  - remote BLEU 更优样本数 = 23 / 200（11.5%）
  - remote ROUGE-L 更优样本数 = 105 / 200（52.5%）
并不好
12. 方案 B：在完成 Plan B（本地先判 + 难例路由远程复判）接线后，使用 `outputs/best.pt` 在课程验收集执行了全量 Plan B 评测（GPU: RTX 5070 Ti），得到：
  - 分类：Acc = 0.6045，AP = 0.7943
  - 解释：BLEU = 0.4124，ROUGE-L = 0.2981
  - 定位：IoU = 0.1501，PixP = 0.3447，PixR = 0.2673，PixF1 = 0.3011
- 新增并验证 Plan B 可观测统计输出（`policy`）：总样本数、路由样本数、远程响应数、override/fallback/human-review 计数与占比、routing reasons 计数、decision source 计数、fallback reasons 计数
- 本次 Plan B 评测统计（classification 子集）显示：
  - total_cases = 402，routed_cases = 81（routing_rate = 20.15%）
  - remote_response_cases = 79（remote_response_rate_over_routed = 97.53%）
  - remote_agreement_rate = 18.99%
  - override_cases = 64（override_rate_over_routed = 79.01%）
  - fallback_cases = 2（fallback_rate_over_routed = 2.47%）
- 本次 Plan B 评测统计（explanation 子集）显示：
  - total_cases = 200，routed_cases = 25（routing_rate = 12.5%）
  - remote_response_cases = 22（remote_response_rate_over_routed = 88.0%）
  - remote_agreement_rate = 13.64%
  - override_cases = 19（override_rate_over_routed = 76.0%）
  - fallback_cases = 3（fallback_rate_over_routed = 12.0%）
- fallback 原因主要包含远程超时与远程 JSON 非法（解析失败），说明线上调用稳定性和输出格式约束仍是实际瓶颈
效果不好
13. 方案 C：- 使用 `distill/run_llm_distill_pipeline.py` 跑通了基于 `qwen3-vl-plus` 的离线蒸馏流程，完成教师结果采集、teacher cache 构建与蒸馏训练，得到蒸馏模型 `outputs_llm_distill/best.pt`
- 在课程验收数据集上分别评估 `outputs_llm_distill/best.pt` 与未蒸馏基线 `outputs/best.pt`，得到：
  - 蒸馏模型：Acc = 0.6766，AP = 0.8036，BLEU = 0.3996，ROUGE-L = 0.2765，IoU = 0.1711，PixP = 0.3640，PixR = 0.3059，PixF1 = 0.3324
  - 基线模型：Acc = 0.6990，AP = 0.8325，BLEU = 0.4618，ROUGE-L = 0.3079，IoU = 0.1501，PixP = 0.3447，PixR = 0.2673，PixF1 = 0.3011
- 对比后确认：当前这版离线蒸馏没有带来整体提升，而是表现出“分类与解释下降、定位上升”的分化结果；相对基线，蒸馏模型的 IoU 与 PixF1 分别提升约 0.0210 和 0.0313，但 Acc、AP、BLEU、ROUGE-L 都出现下降
- 结合训练与蒸馏实现进一步分析后，初步判断当前蒸馏效果不升反降的主要原因包括：教师覆盖率偏低（teacher 仅采集了约 1000 张，而 Stage B 联合训练集约 3 万张）、分类 teacher 仅由 LLM 输出的标签和置信度压缩成两类伪 logits、explanation teacher 文本分布与课程 explanation 标注分布不一致、且当前蒸馏只在 Stage B 生效，更容易偏向提升定位相关分支
 