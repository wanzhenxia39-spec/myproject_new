# 五数据集 A1 初始属性向量批量生成

入口为 `scripts/generate_initial_vectors_all.py`。默认依次生成 PAMAP2、MHEALTH、
USC-HAD（内部键 `uschad`）、WISDM 和 DSADS，使用智增增 OpenAI-compatible
端点及 `gpt-5.6-sol`。

## 提示信息边界

A1 是数据集级、split 之前的纯语义基线。LLM 只接收：

- 正式九维 A1 ontology（名称、定义、Low/Medium/High）；
- 完整活动名单；
- 数据集特定的易混活动比较重点及必要的活动名称说明；
- 统一的输出格式和语义一致性规则。

提示中不包含 G1–G14、具体特征模板、属性—特征组映射、传感器清单、特征值、
样本、subject 或 seen/unseen split。属性到可执行特征组的绑定只在后续 A2 细化阶段
处理。报告中的策略标识为 `semantic_a1_without_feature_groups_v2`。

## 完整操作

```bash
cd /home/wanzhen/myproject_new
export PYTHONPATH="$PWD/src"

read -rsp "Zhizengzeng API key: " ZZZ_API_KEY
export ZZZ_API_KEY
echo

# 短请求验证密钥、端点和模型。
curl -sS https://api.zhizengzeng.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ZZZ_API_KEY" \
  -d '{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"Reply with OK only."}],"temperature":0}' \
  | python -m json.tool

# 检查五个将执行的命令；不调用 API、不改文件。
/home/wanzhen/miniconda3/envs/zara/bin/python \
  scripts/generate_initial_vectors_all.py --dry-run

# 依次生成、严格校验并安装五份最新 A1。
/home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/generate_initial_vectors_all.py \
  --base-url https://api.zhizengzeng.com/v1 \
  --model gpt-5.6-sol \
  --stream --timeout 600 \
  --max-attempts 5 --retry-delay 60 --retry-backoff 1.5
```

批处理默认把 `api.zhizengzeng.com` 加入子进程的 `NO_PROXY`，避免本机
`HTTP(S)_PROXY` 引起 `SSL: UNEXPECTED_EOF_WHILE_READING`。若所在网络必须通过
代理访问智增增，可显式加 `--use-proxy`。

旧向量、prompt、原始响应和报告会先备份到：

```text
runtime/a1_generation_zzz_gpt_5_6_sol/backups/<timestamp>/
```

日志和批次状态保存在：

```text
runtime/a1_generation_zzz_gpt_5_6_sol/logs/
runtime/a1_generation_zzz_gpt_5_6_sol/batch_manifest.json
```

若流式接口异常，将 `--stream` 换为 `--no-stream`。只重跑部分数据集可使用：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/generate_initial_vectors_all.py \
  --datasets mhealth,uschad \
  --base-url https://api.zhizengzeng.com/v1 \
  --model gpt-5.6-sol --stream
```

五份正式输出为 `configs/attributes/activity_attribute_vectors_initial_<dataset>_v1_5.json`。
同目录同时保存 `.prompt.json`、`.prompt.txt`、`.raw_response.txt` 和 `.report.json`。
只有活动键、九维属性键、三级取值、生成策略、ontology/context 哈希及无重复完整向量
全部通过检查时，该数据集才记为 `COMPLETE`。

完成后可清除当前 shell 中的密钥：

```bash
unset ZZZ_API_KEY
```
