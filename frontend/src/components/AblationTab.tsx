import { useState } from 'react';
import { MODELS_BENCHMARK, ModelId, TARGET_CLASSES } from '../data/benchmarkData';

export function AblationTab() {
  const [selectedId, setSelectedId] = useState<ModelId>('a5_opt');
  const [rightView, setRightView] = useState<'matrix' | 'f1' | 'png'>('matrix');

  const selected = MODELS_BENCHMARK.find((m) => m.id === selectedId)!;
  const maxCell = Math.max(1, ...selected.confusion_matrix.flat());

  return (
    <div className="two-col">
      {/* Left: Concise Ablation Table */}
      <div className="card">
        <h2 className="section-heading">Ablation Results</h2>
        <p className="section-sub">
          Evaluated on 933 TrashCan-1.0 validation images. Click a row to inspect.
        </p>
        <div className="table-wrap">
          <table className="clean">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>Weighted F1</th>
              </tr>
            </thead>
            <tbody>
              {MODELS_BENCHMARK.map((m) => (
                <tr
                  key={m.id}
                  onClick={() => setSelectedId(m.id)}
                  className={m.id === selectedId ? 'active-row' : ''}
                  style={{ cursor: 'pointer' }}
                >
                  <td style={{ fontWeight: 600 }}>{m.name}</td>
                  <td className="mono">{m.val_accuracy.toFixed(2)}%</td>
                  <td className="mono">{m.macro_f1.toFixed(4)}</td>
                  <td className="mono">{m.weighted_f1.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--ink-muted)', marginTop: '0.9rem', marginBottom: 0 }}>
          {selected.key_finding}
        </p>
      </div>

      {/* Right: Confusion Matrix / Per-Class F1 */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.9rem' }}>
          <h2 className="section-heading" style={{ margin: 0 }}>{selected.name}</h2>
          <div className="segmented">
            <button className={rightView === 'matrix' ? 'active' : ''} onClick={() => setRightView('matrix')}>
              Matrix
            </button>
            <button className={rightView === 'f1' ? 'active' : ''} onClick={() => setRightView('f1')}>
              Per-Class
            </button>
            <button className={rightView === 'png' ? 'active' : ''} onClick={() => setRightView('png')}>
              PNG
            </button>
          </div>
        </div>

        {rightView === 'matrix' && (
          <div className="table-wrap">
            <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '3px', fontSize: '0.75rem' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', color: 'var(--ink-subtle)', padding: '0.25rem' }}>True \ Pred</th>
                  {TARGET_CLASSES.map((c) => (
                    <th key={c} style={{ textAlign: 'center', color: 'var(--ink-muted)', padding: '0.25rem' }}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {selected.confusion_matrix.map((row, r) => (
                  <tr key={TARGET_CLASSES[r]}>
                    <td style={{ fontWeight: 600, padding: '0.3rem', color: 'var(--ink-muted)' }}>{TARGET_CLASSES[r]}</td>
                    {row.map((val, c) => {
                      const alpha = val === 0 ? 0 : 0.16 + (val / maxCell) * 0.72;
                      const bg = val === 0 ? 'var(--alabaster)' : r === c ? `rgba(24, 27, 23, ${alpha})` : `rgba(153, 27, 27, ${alpha})`;
                      const color = alpha > 0.45 ? '#ffffff' : 'var(--ink)';
                      return (
                        <td
                          key={`${r}-${c}`}
                          className="mono"
                          style={{
                            textAlign: 'center',
                            padding: '0.5rem 0.25rem',
                            borderRadius: '6px',
                            background: bg,
                            color,
                            fontWeight: val > 0 ? 600 : 400,
                          }}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {rightView === 'f1' && (
          <div className="table-wrap">
            <table className="clean">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {selected.per_class.map((r) => (
                  <tr key={r.name}>
                    <td style={{ fontWeight: 600 }}>{r.name}</td>
                    <td className="mono">{r.precision.toFixed(4)}</td>
                    <td className="mono">{r.recall.toFixed(4)}</td>
                    <td className="mono" style={{ fontWeight: r.f1 >= 0.9 ? 700 : 400 }}>{r.f1.toFixed(4)}</td>
                    <td className="mono">{r.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {rightView === 'png' && (
          <img
            src={selected.notebook_cm_url}
            alt={`Confusion matrix for ${selected.name}`}
            style={{ width: '100%', height: 'auto', borderRadius: '8px' }}
          />
        )}
      </div>
    </div>
  );
}
