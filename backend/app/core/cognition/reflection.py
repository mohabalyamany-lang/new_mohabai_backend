def reflect_on_output(answer: str, intent: str) -> dict:
    """
    Self-evaluation layer.
    Determines if response satisfies intent.
    """

    issues = []

    if not answer or len(answer.strip()) < 5:
        issues.append("empty_or_short")

    if intent == "image_edit" and "cannot" in answer.lower():
        issues.append("failed_tool_usage")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }
