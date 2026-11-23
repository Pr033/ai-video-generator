# 剧本 A/B 对比
Python 3.10+。安装 python -m pip install "langchain>=1,<2" requests 。设置 CHAT_BASE_URL（含 /v1 或 /api/v3）、CHAT_MODEL、CHAT_API_KEY。
运行 python compare.py "雨后校园，最多18秒，结尾慢推，不要旁白" --slots 5 5 5 --out run01
每次使用新的输出目录。两版共用一次需求分析，模型和采样温度相同。baseline.json 和 vimax.json 均可交给分镜；vimax_raw.json 是适配前的原稿。comparison.json 记录编剧阶段请求数、服务端返回的 token 数和耗时，不含共享需求分析。缺失的用量数据按零统计，不能据此推算费用。
主控可直接导入 compare；单独使用 ViMax 时在 generate 的 model_call 回调里把 WRITE 分支交给 write_vimax(payload, chat)[0]，其他分支用 chat.structured。同步入口请在后台线程调用。
上游 HKUDS/ViMax commit fd4b72e7731be6e5f88486206d25ea59849b0285。agents/screenwriter.py、utils/retry.py、utils/robust_json_parser.py 取自该提交，许可证见 VIMAX_LICENSE。仅运行文本编剧。质量评审建议隐藏文件名，从主题、动作可拍性、运镜、时长可行性各给 0–2 分。
单后端入口：from single import generate_script，然后 generate_script(idea, slots=[5,5,5], backend="vimax")。
