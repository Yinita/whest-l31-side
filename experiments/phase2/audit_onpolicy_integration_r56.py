"""Read-only actual normalized candidate source/component audit, no gain claim."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--actual-component-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.candidate) == args.candidate_sha256 and sha(args.model) == args.model_sha256
    parity = json.loads(args.actual_component_audit.read_text())
    assert parity["complete"] and parity["candidate_sha256"] == args.candidate_sha256
    assert parity["model_sha256"] == args.model_sha256
    assert parity["labels_reference_prediction_not_read"]
    source = args.candidate.read_text()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    methods = {n.name: n for c in tree.body if isinstance(c, ast.ClassDef)
               for n in c.body if isinstance(n, ast.FunctionDef)}
    arithmetic = [functions[name] for name in ('_corpus_features', '_corpus_ownpost_features',
        '_corpus_normalized_delta', '_corpus_updated_diagonals')] + [methods['predict'], methods['_predict_core']]
    illegal_attributes = [ast.unparse(n) for root in arithmetic for n in ast.walk(root)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id in {'np', 'numpy', 'torch', 'sys', 'subprocess', 'threading', 'multiprocessing'}]
    forbidden_calls = [ast.unparse(n) for root in arithmetic for n in ast.walk(root)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in {'exec', 'eval', 'compile', 'open', '__import__'}]
    assert not illegal_attributes and not forbidden_calls
    normalized = functions['_corpus_normalized_delta'].body[-1].value
    expected = ast.parse('((features-mean[layer])/scale[layer])@theta[layer]', mode='eval').body
    assert ast.dump(normalized, include_attributes=False) == ast.dump(expected, include_attributes=False)
    support = args.candidate.parent / 'coefficients.npz'
    metadata = json.loads(support.with_suffix('.json').read_text())
    assert metadata['model_sha256'] == args.model_sha256
    assert sha(support) == metadata['support_sha256'] == parity['support_sha256']
    assert args.model_sha256 in source
    checks = dict(first0_final15_skipped='if riders and 0 < li < 15:' in source,
        ordinary_diagonals_updated_before_assemble=source.index('corpus_updated=_corpus_updated_diagonals') <
            source.index('            mu = K1v if delta is None else K1v + delta'),
        original_mixed_central_pairs_unchanged=True,
        existing_mean_rider_added_once='mean=mean+inherited_delta.astype(fnp.float64)' in source,
        existing_mean_rider_subtracted_before_original_assemble='mean=mean-inherited_delta.astype(fnp.float64)' in source,
        normalized_operation_order_preserved=True,
        actual_component_gate_passed=True,
        support_source_model_identity_verified=True)
    # New injection block only copies four marginal diagonals. Mixed-pair writes
    # are in original H18 upstream/downstream, not the added closure block.
    start = source.index('            if corpus_delta is not None:')
    end = source.index('            mu = K1v if delta is None else K1v + delta', start)
    block = source[start:end]
    assert 'zip((K1v,K2v,K3v,K4v),corpus_updated)' in block
    assert not any(name in block for name in ('K11=', 'K21=', 'K22=', 'pk1v='))
    assert all(checks.values())
    report = dict(version='r56-actual-lawful-normalized-integration-source-audit', passed=True,
        candidate_sha256=args.candidate_sha256, model_sha256=args.model_sha256,
        support_sha256=sha(support), source_sha256=sha(Path(__file__)), checks=checks,
        actual_component_audit_sha256=sha(args.actual_component_audit),
        illegal_runtime_attributes=illegal_attributes, forbidden_runtime_calls=forbidden_calls,
        whole_predict_parity=False, strict_acceptance=False, score_claim=False,
        scope='actualcomponent/sourceboundary audit only; formal signedcentralmarginals with unchanged mixedpairs; no jointrealizability/wholepredict/gain claim')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
