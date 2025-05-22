import re, ast, operator, math

# ---------- expression evaluator ---------- #
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
def _safe_eval(expr: str) -> float:
    """Safely evaluate a numeric expression using the AST module."""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):         # literal number
            return node.n
        if isinstance(node, ast.UnaryOp):     # -something
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp):       # a + b, a * b …
            if type(node.op) not in _ALLOWED_OPS:
                raise ValueError("Operator not allowed")
            return _ALLOWED_OPS[type(node.op)](
                _eval(node.left), _eval(node.right)
            )
        raise TypeError(node)

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)

# ---------- main reward function ---------- #
def calculate_score(prompt: str, response: str) -> float:
    """
    Countdown reward:
      1.0  → exact solution
      0.0–1 → proportional to closeness   (1 - |diff| / target), clipped to [0,1]
      0.0  → malformed / wrong numbers / unsafe expr
    """
    try:
        # Parse prompt → target + list of ints
        target_match = re.search(r"Target\s*=\s*(-?\d+)", prompt, re.I)
        bracket_match = re.search(r"\[([0-9,\s]+)\]", prompt)

        if not target_match or not bracket_match:
            return 0.0

        target = int(target_match.group(1))
        numbers = list(map(int, bracket_match.group(1).split(",")))

        # Extract first arithmetic expression from the response
        expr_match = re.search(r"[0-9(][0-9\s+\-*/().]*", response)
        if not expr_match:
            return 0.0
        expr = expr_match.group(0).replace(" ", "")

        # Ensure only safe characters
        if not re.fullmatch(r"[0-9+\-*/().]+", expr):
            return 0.0

        # Evaluate
        result = _safe_eval(expr)
        if not math.isfinite(result):
            return 0.0

        # Check numbers used (each at most once)
        allowed = numbers.copy()
        for n in map(int, re.findall(r"\d+", expr)):
            if n not in allowed:
                return 0.0            # unlisted OR reused
            allowed.remove(n)

        # Scoring
        diff = abs(result - target)
        if diff == 0:
            return 1.0
        return max(0.0, 1.0 - diff / target)

    except Exception:
        return 0.0
