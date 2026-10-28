# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Reproducible text tasks, with local tokenizer and no private business data."""
import hashlib
import json


def build_workloads(model, cycles):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    prompts = {}

    def register(name, ids, task, expected=None):
        prompts[name] = dict(name=name, token_ids=ids, input_tokens=len(ids), task=task,
                             expected=expected, rendered_prompt=tokenizer.decode(ids),
                             sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        return prompts[name]

    def chat(name, text, expected=None):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                             tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False)
        return register(name, ids, text, expected)

    extraction = chat('extract-json',
        '请从以下虚构工单抽取字段，严格只输出一个 JSON 对象，不要解释。'
        '字段为 ticket_id、city、device_count、status。'
        '工单记录：编号 SR-2026-0816；城市：杭州；申请设备数量：12；'
        '初审已通过，当前状态为等待交付。联系人字段不要输出。',
        {'ticket_id': 'SR-2026-0816', 'city': '杭州', 'device_count': 12, 'status': '等待交付'})
    code = chat('python-code',
        '请编写 Python 函数 merge_intervals(intervals)，合并重叠或端点相接的闭区间。'
        '要求不修改输入，空列表返回空列表，输出按左端点排序。'
        '仅输出函数代码和三个 assert 示例，覆盖空输入、嵌套区间和端点相接；不要解释。')
    summary = chat('chinese-summary',
        '将以下虚构会议记录整理成三条中文行动项，每条写明负责人、动作、截止时间。不要添加事实。'
        '周一研发例会：陈工周三前补齐长上下文推理测试，覆盖 1K、4K、8K 输入。'
        '李工周四前核对监控中的物理显存和逻辑缓存指标。王工周五前复查服务退出后是否残留进程。'
        '暂不开展自动调度，先验证现有单卡路径。')

    def long(name, length, key, value):
        question = (f'下面是虚构的仓储日志。请找到查询编号 {key} 对应的校验码，'
                    '只输出校验码，不要输出推理过程。\n')
        marker = 'SYD_RECORDS_BODY_20261029'
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question+marker+'\n日志结束。请只回答查询编号 '+key+' 的校验码。'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert rendered.count(marker) == 1
        before, after = rendered.split(marker)
        head = tokenizer.encode(before, add_special_tokens=False)
        tail = tokenizer.encode(after, add_special_tokens=False)
        needle = tokenizer.encode('\n查询编号：'+key+'；校验码：'+value+'。\n', add_special_tokens=False)
        filler = ''.join(f'记录 {i:05d}：站点 {(i*17)%997:03d}，货架 {(i*7)%113:03d}，'
                         f'库存 {(i*11)%71+1} 箱，状态已核验。\n' for i in range(1800))
        body = tokenizer.encode(filler, add_special_tokens=False)
        budget = length-len(head)-len(tail)-len(needle)
        assert 0 < budget < len(body)
        split = budget*2//3
        ids = head+body[:split]+needle+body[split:budget]+tail
        assert len(ids) == length
        return register(name, ids, question, value)

    needle1 = long('retrieve-1k', 1024, 'TASK-1703', '海风-7284')
    needle4 = long('retrieve-4k', 4096, 'TASK-2906', '青竹-5917')
    needle8 = long('retrieve-8k', 8192, 'TASK-4308', '晨光-3629')
    burst = [long(f'retrieve-burst-{i}', 7168, f'BATCH-{5000+i}', f'RESULT-{6100+i}')
             for i in range(4)]
    mixed = [extraction, needle1, needle4, needle8]
    workloads = [('extract-json', [extraction], 128), ('python-code', [code], 256),
                 ('chinese-summary', [summary], 192), ('retrieve-1k', [needle1], 64),
                 ('retrieve-4k', [needle4], 64), ('retrieve-8k', [needle8], 64)]
    for cycle in range(cycles):
        workloads.extend([(f'mixed-{cycle}', mixed, 128), (f'burst-{cycle}', burst, 64)])
    return workloads, [needle8], burst, prompts
