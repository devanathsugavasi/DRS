from core.lbw_engine import LBWDecisionEngine


def test_out_in_line_hitting_stumps():
    decision = LBWDecisionEngine().evaluate(0, 0, 350, 0.9, 0.9)
    assert decision.verdict == "OUT"


def test_not_out_pitched_outside_leg():
    decision = LBWDecisionEngine().evaluate(-200, 0, 350, 0.9, 0.9)
    assert decision.verdict == "NOT_OUT"


def test_umpires_call_clipping_margin():
    decision = LBWDecisionEngine().evaluate(0, 0, 350, 0.52, 0.9)
    assert decision.verdict == "UMPIRE_CALL"


def test_inconclusive_when_missing_data():
    decision = LBWDecisionEngine().evaluate(None, 0, 350, 0.9, 0.9)
    assert decision.verdict == "REVIEW_INCONCLUSIVE"
