import json
import numpy as np
import pandas as pd
import pytest
from tennis_ai.simulate import make_demo_sessions
from tennis_ai.io import validate_sessions, load_sessions
from tennis_ai.features import derive_session_metrics, prior_only_features, compare_recent_blocks, trend_per_10_sessions
from tennis_ai.model import walk_forward_predictions, evaluate_predictions, leaky_same_match_auc
from tennis_ai.assistant import build_evidence_packet, build_llm_prompt, deterministic_brief


def test_demo_schema_validates():
    d=validate_sessions(make_demo_sessions(40))
    assert len(d)==40 and d.session_id.is_unique


def test_invalid_count_fails():
    d=make_demo_sessions(10); d.loc[0,"first_serves_in"]=d.loc[0,"serve_points"]+1
    with pytest.raises(ValueError): validate_sessions(d)


def test_duplicate_session_fails():
    d=make_demo_sessions(10); d.loc[1,"session_id"]=d.loc[0,"session_id"]
    with pytest.raises(ValueError): validate_sessions(d)


def test_derived_rates_are_bounded():
    x=derive_session_metrics(make_demo_sessions(30))
    for c in ["first_serve_pct","forehand_in_pct","backhand_in_pct","rallies_5plus_pct"]:
        assert x[c].between(0,1).all()


def test_prior_features_do_not_use_current_session():
    d=make_demo_sessions(40)
    a=prior_only_features(d)
    d2=d.copy(); d2.loc[20,"first_serves_in"]=0; d2.loc[20,"result"]="L" if d2.loc[20,"result"]=="W" else "W"
    b=prior_only_features(d2)
    # current row 20 features are based only on <=19
    cols=[c for c in a.columns if c.startswith("recent_")]
    pd.testing.assert_series_equal(a.loc[20,cols],b.loc[20,cols])


def test_future_perturbation_cannot_change_past_features():
    d=make_demo_sessions(50); a=prior_only_features(d)
    d2=d.copy(); d2.loc[45:,"first_serves_in"]=0; d2.loc[45:,"winners"]=999
    b=prior_only_features(d2)
    cols=[c for c in a.columns if c.startswith("recent_")]
    pd.testing.assert_frame_equal(a.loc[:44,cols],b.loc[:44,cols])


def test_recent_block_comparison():
    out=compare_recent_blocks(make_demo_sessions(40),block=8)
    assert {"metric","previous_mean","recent_mean","change"}.issubset(out.columns)


def test_known_truth_progress_is_recovered_directionally():
    d=make_demo_sessions(120)
    assert trend_per_10_sessions(d,"avg_serve_speed_kmh") > 0
    assert trend_per_10_sessions(d,"backhand_in_pct") > 0


def test_walk_forward_is_probabilistic_and_nonempty():
    pred=walk_forward_predictions(make_demo_sessions(100),min_train=25)
    assert len(pred)>40 and pred.p_model.between(0,1).all()


def test_walk_forward_metrics_finite():
    m=evaluate_predictions(walk_forward_predictions(make_demo_sessions(100),min_train=25))
    for k in ["brier_model","brier_baseline","logloss_model","logloss_baseline","auc_model"]:
        assert np.isfinite(m[k])


def test_leaky_diagnostic_is_only_posthoc_and_strong_in_demo():
    auc=leaky_same_match_auc(make_demo_sessions(120))
    assert auc > 0.80


def test_evidence_packet_has_no_raw_opponent_names():
    p=build_evidence_packet(make_demo_sessions(40))
    text=json.dumps(p)
    assert "Opponent" not in text


def test_llm_prompt_has_grounding_guardrails():
    p=build_evidence_packet(make_demo_sessions(40))
    s=build_llm_prompt(p)
    assert "ONLY the evidence" in s and "Do not invent" in s


def test_deterministic_brief_is_descriptive():
    b=deterministic_brief(build_evidence_packet(make_demo_sessions(40)))
    assert "descriptive" in b.lower() and "causal" in b.lower()


def test_csv_roundtrip(tmp_path):
    d=make_demo_sessions(20); p=tmp_path/"sessions.csv"; d.to_csv(p,index=False)
    got=load_sessions(p)
    assert len(got)==20


def test_column_map(tmp_path):
    d=make_demo_sessions(20).rename(columns={"date":"Date"}); p=tmp_path/"x.csv"; d.to_csv(p,index=False)
    got=load_sessions(p,{"Date":"date"})
    assert len(got)==20


def test_promotion_gate_blocks_demo_when_probability_score_is_worse():
    from tennis_ai.model import promotion_gate
    m=evaluate_predictions(walk_forward_predictions(make_demo_sessions(100),min_train=25))
    g=promotion_gate(m,min_predictions=40)
    assert g["promote"] is False
    assert any("Brier" in r for r in g["reasons"])


def test_workbook_inspector_returns_schema_not_values(tmp_path):
    from tennis_ai.inspect_export import inspect_workbook
    p=tmp_path/"private.xlsx"
    pd.DataFrame({"SecretName":["Alice"],"ServeSpeed":[123.4]}).to_excel(p,sheet_name="Shots",index=False)
    out=inspect_workbook(p)
    text=json.dumps(out)
    assert "SecretName" in text and "ServeSpeed" in text
    assert "Alice" not in text and "123.4" not in text
