from titans.stage1_prompting import build_stage1_question_prompt, default_stage1_checkpoint_dir


def test_prompt_versions_render_different_templates() -> None:
    q = "张三住在哪？"
    v2 = build_stage1_question_prompt(q, "v2")
    v3 = build_stage1_question_prompt(q, "v3")
    v4 = build_stage1_question_prompt(q, "v4")
    assert v2 == "问题：张三住在哪？\n答案："
    assert "请只输出最终答案，不要解释。" in v3
    assert v3.endswith("回答：")
    assert "最相关的历史要点" in v4
    assert "请只输出最终答案，不要解释。" in v4
    assert v4.endswith("最终答案：")


def test_checkpoint_dir_follows_prompt_version() -> None:
    assert default_stage1_checkpoint_dir("v2") == "checkpoints/stage1_timeline_v2"
    assert default_stage1_checkpoint_dir("v3") == "checkpoints/stage1_timeline_v3"
    assert default_stage1_checkpoint_dir("v4") == "checkpoints/stage1_timeline_v4"
