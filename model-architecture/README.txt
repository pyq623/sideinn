模型架构专栏（静态页面）

线上入口：https://sideinn.ink/model-architecture/

入口：index.html
Qwen3.6：qwen36/index.html
Qwen3-30B-A3B：qwen3-30b-a3b/index.html
Qwen3-8B：qwen3-8b/index.html

修改 Qwen3 两款图：build-qwen3.py（坐标、连线、文案）与 drawing.py（SVG 基元）。
修改两款阅读页：qwen3-template.html、qwen3-viewer.css、qwen3-viewer.js。
从站点根目录执行：python3 model-architecture/build-qwen3.py
生成两个模型各自的 index.html 和 replica / improved SVG。
config.snapshot.json 保存本次核对的官方配置，sources.json 记录来源与适用版本。
HTML 模板中的占位符供构建使用，不应直接作为阅读页打开。

Qwen3.6 独立维护：python3 model-architecture/qwen36/build.py
首页卡片：../assets/notebook.json；专栏卡片：index.html。
图表中的 ×N 表示重复结构，各层参数独立，不代表权重共享。
复刻版为参考原图构图的矢量重绘；改进版按官方配置和实现展开。

语言约定（用户偏好）：模块名与专有术语保留英文，如 Gated DeltaNet、RMSNorm、RoPE、MoE、SwiGLU；流程说明、参数含义、重复范围、图例及来源注释使用中文。复刻版与改进版均遵循此约定。

Ling-2.6-flash（Bailing v2.5）：ling-2.6-flash/index.html
构建：python3 model-architecture/build-ling.py；模板：ling-2.6-flash/template.html。
参考布局按官方 1T 图适配 Flash；完整展开图依据固定版本 config.snapshot.json 与 implementation.snapshot.py。来源与提交见 sources.json。32 层 Decoder 主干，不含辅助 MTP。

术语对应约定：图中模块标签与中文解释中的专有名词必须逐字一致（含大小写），不得部分翻译或换用缩写。例如图中 Up projection，解释也写 Up projection，不写 Up 投影或上投影；Gate projection、Down projection 同理。中文仅用于解释其作用与数据流。

第三方来源：Ling 的 implementation.snapshot.py 为官方代码快照，保留原始版权声明；对应 Apache-2.0 许可全文见 ling-2.6-flash/LICENSE-implementation.txt。页面仅引用该快照作资料依据，不在浏览器中执行。
