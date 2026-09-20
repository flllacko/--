from __future__ import annotations

import re
from typing import Literal


SearchScope = Literal["current", "historical", "navigation"]


def _normalized(question: str) -> str:
    return re.sub(r"\s+", "", question).strip()


def is_explicit_historical_query(question: str) -> bool:
    """识别用户明确要求复述历史批次，而不是把旧资料当作当前政策。"""

    normalized = _normalized(question)
    # 四川省专升本专业对应目录按年度发布；问“2025/2026/2027 能否直接套用”
    # 时必须进入年度资料范围，不能被当前转专业细则等相近页面干扰。
    if "专升本" in normalized and "对应目录" in normalized:
        return True
    if any(marker in normalized for marker in ("历史", "往年", "上一届", "最近一次公开")):
        return True
    if "按" in normalized and any(
        marker in normalized for marker in ("附件", "流程", "指南", "办法", "制度", "通知", "下载页", "管理办法", "政策")
    ):
        return True
    if "2026级" in normalized and any(
        marker in normalized for marker in ("选宿舍", "床位", "床上用品", "卧具", "退宿")
    ):
        return True
    # 2026 届论文工作计划目前有一份标记为有效的汇总正文，不能因为“届”字
    # 直接切到历史范围；论文问题由资料自己的有效性和年份边界控制。毕业离校、
    # 退宿等已结束批次事务仍保留历史路由。
    if "2026届" in normalized and any(
        marker in normalized for marker in ("退宿", "离校", "一辩", "二辩", "答辩前", "迟到")
    ):
        return True
    if "2025级" in normalized and any(marker in normalized for marker in ("转入", "转专业", "面试", "申请材料")):
        return True
    # 2026 年的正式招生计划、学费和专升本培养规则属于当前问答范围；
    # 只有明确问到当年的报考指南、录取结果类历史批次资料时才切到历史范围。
    if re.search(r"2026(?:年)?", normalized) and any(
        marker in normalized
        for marker in ("报考指南", "录取查询", "录取分数", "打印准考证", "退役士兵批")
    ):
        return True
    if "2026" in normalized and "专升本" in normalized and any(
        marker in normalized for marker in ("退役大学生士兵", "免试", "考生类别")
    ):
        return True
    if "专升本" in normalized and not any(
        marker in normalized
        for marker in ("当前", "现在", "本学期", "2026", "2026—2027", "2026-2027")
    ):
        return True
    # 图书借阅、座位预约对应当前图书馆积分规则；不能被泛化的“第二课堂”
    # 历史政策规则抢走。其他未指明业务的第二课堂问题仍按历史资料回答。
    if any(marker in normalized for marker in ("第二课堂", "到梦空间")) and not any(
        marker in normalized for marker in ("图书借阅", "座位预约")
    ):
        return True
    if "6—7月" in normalized and "助学金" in normalized:
        return True
    if "乐山站" in normalized or "交通路线" in normalized:
        return True
    if "宿舍访客" in normalized or ("访客" in normalized and "公寓" in normalized):
        return True
    if "体育系" in normalized and any(marker in normalized for marker in ("晚归", "夜不归宿", "走读")):
        return True
    if "食品安全" in normalized and any(marker in normalized for marker in ("事故", "疑似", "处置")):
        return True
    # 只有明确引用学校公开说明、询问报警后的处置责任时，才查历史演练/通告；
    # “无故鸣叫或损坏怎么办”应命中当前报修说明。
    if "烟雾报警器" in normalized and (
        "学校公开说明" in normalized
        or "报警后" in normalized
        or "谁会处置" in normalized
        or "谁会到场" in normalized
    ):
        return True
    if "宿舍" in normalized and any(marker in normalized for marker in ("突发急病", "停水停电", "公开流程")):
        return True
    # 这组资料是 2026 年招生咨询 FAQ，适合回答专业课程、难度、导师、
    # 竞赛和就业方向等专业介绍问题；不把它们当作当前招生计划或收费规则。
    program_names = ("智能建造", "城乡规划", "安全工程", "工程造价", "土木工程")
    program_question_markers = (
        "课程",
        "学习",
        "难度",
        "导师",
        "竞赛",
        "就业",
        "毕业",
        "工作",
        "技能",
        "考研",
        "方向",
    )
    if (
        any(program in normalized for program in program_names)
        and any(marker in normalized for marker in program_question_markers)
        and not any(marker in normalized for marker in ("当前", "现在", "2027", "2028"))
    ):
        return True

    # 这些问题需要复述已经公开的年度/批次规则。显式切到历史范围，
    # 让当前范围不会用相近但不适用的页面替代确切答案。
    if "2027届" in normalized and "生源信息" in normalized:
        return True
    if "专升本" in normalized and "现代文秘" in normalized and any(
        marker in normalized for marker in ("对应", "考试科目", "本科专业")
    ):
        return True
    if "国家助学金" in normalized:
        years = [int(value) for value in re.findall(r"20\d{2}", normalized)]
        if years and max(years) < 2026:
            return True
        if "去年" in normalized and "今年" in normalized:
            return True
        if any(marker in normalized for marker in ("休学", "保留学籍", "退学")) and not any(
            marker in normalized for marker in ("当前", "现在", "本学年", "本学期")
        ):
            return True
    if "国家奖学金" in normalized and "国家励志奖学金" in normalized and not any(
        marker in normalized for marker in ("当前", "现在", "本学年", "本学期", "2026—2027", "2026-2027")
    ):
        return True
    if "家庭经济困难认定" in normalized and any(marker in normalized for marker in ("去年", "今年", "重新申请")):
        return True
    if "勤工助学" in normalized and any(marker in normalized for marker in ("在岗", "报名", "录用")) and not any(
        marker in normalized for marker in ("当前", "现在", "本学期", "2026—2027", "2026-2027")
    ):
        return True
    # 这是 2026 年官方政策图文明确覆盖的分段规则。问题中的“2024 年 9 月 1 日以后”
    # 是政策适用起点，不代表用户在追问一份 2024 年历史通知。
    if (
        "2024年9月1日" in normalized
        and ("以后" in normalized or "之后" in normalized)
        and ("入伍" in normalized or "服兵役" in normalized)
        and ("学费补偿" in normalized or "贷款代偿" in normalized)
    ):
        return False
    years = [int(value) for value in re.findall(r"20\d{2}", normalized)]
    if not years or max(years) >= 2026:
        return False
    return any(
        marker in normalized
        for marker in ("按", "公开", "指南", "附件", "办法", "流程", "通知", "档案", "届", "级")
    )


def is_navigation_query(question: str) -> bool:
    """识别只要求招生公开入口或查询页面的问题。"""

    normalized = _normalized(question)
    if "专业设置页" in normalized:
        return True
    return "录取查询系统" in normalized and any(
        marker in normalized for marker in ("入口", "在哪里", "在哪看", "怎么查")
    )


def resolve_scope(question: str, requested_scope: SearchScope) -> SearchScope:
    """默认当前范围；用户明确点名历史材料时自动切到历史范围。"""

    if requested_scope != "current":
        return requested_scope
    if is_navigation_query(question):
        return "navigation"
    return "historical" if is_explicit_historical_query(question) else "current"


def is_known_current_gap(question: str) -> bool:
    """拦截已核实的当前公开资料缺口，防止相近页面被扩大解释。"""

    normalized = _normalized(question)
    if is_explicit_historical_query(normalized):
        return False

    current_hint = any(
        marker in normalized
        for marker in ("当前", "现在", "本学年", "本学期", "2026—2027", "2026-2027", "我校")
    )
    if "寝室几点熄灯" in normalized or "宿舍几点熄灯" in normalized:
        return True
    if ("熄灯" in normalized or "住宿门禁" in normalized) and current_hint:
        return True
    if ("调宿" in normalized or "分配床位" in normalized) and current_hint:
        return True
    if (
        "住宿费" in normalized
        and "房型" in normalized
        and current_hint
        and not ("阳光高考" in normalized and "当前展示" in normalized)
    ):
        return True
    if "国家助学金" in normalized and (
        "评审" in normalized or "发放" in normalized or "到账" in normalized or "什么时候发" in normalized
    ) and (current_hint or not re.search(r"20\d{2}", normalized)):
        return True
    if all(term in normalized for term in ("休学", "复学", "退学")) and (
        "完整" in normalized or "办理时限" in normalized
    ) and current_hint:
        return True
    return False
