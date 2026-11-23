# Trae 剧本 Agent
解压并在 Trae 打开包含 main.py 的文件夹，选择 Python 3.11 或 3.12 解释器。终端运行 python main.py --demo 可直接查看已有样例。
真实生成：先在该解释器运行 python -m pip install -r requirements.txt，再编辑项目内的 .env，填入模型 ID 和 API Key；其他兼容服务请修改 CHAT_BASE_URL。支持 Trae 的 Python 调试环境时可选运行配置；终端入口始终可用。
运行 python main.py "雨后校园，两人和解，最多18秒，结尾慢推，无旁白" --budget ample --slots 5 5 5
默认 auto + standard。auto 在 ample 且需求分析判断为复杂叙事时用 ViMax，其他情况用 baseline。--backend baseline 或 --backend vimax 可强制选择。预算档由用户提供，程序不读取余额，不保证费用上限。ViMax 用于文本编剧。
结果在 outputs/，每次独立保存。ready 时下游读取 brief 和 beats；needs_input 时读取 question。slots 可由主控传入后端实际支持的整数秒时段。
主控集成：from main import run_agent；result = run_agent(idea, slots=[5,5,5], budget="ample")。routing 说明选择原因，usage 记录本次请求数和服务端报告的 token。调用失败向主控抛出异常。
A/B 对比：先在终端导出 CHAT_* 环境变量，或用 python -c "from main import load_config; load_config(); from compare import compare; print(compare('雨后校园，最多18秒','run01',[5,5,5]))" 。使用新目录名。两组共用需求单但分别产生文本费用。sample/ 保存之前的真实对照。来源和许可证见 AB_README.md 和 VIMAX_LICENSE。
