import { ExternalLink, RefreshCw } from 'lucide-react';
import { HealthStatus } from '../api/client';
import { assetUrl } from '../data/benchmarkData';

interface ReportTabProps {
  health: HealthStatus | null;
  onRefreshHealth: () => void;
}

export function ReportTab({ health, onRefreshHealth }: ReportTabProps) {
  return (
    <div className="stack">
      <div className="card" style={{ background: 'var(--sage-tint)' }}>
        <h2 className="section-heading">Project Report & Deliverables</h2>
        <p className="section-sub" style={{ maxWidth: '700px' }}>
          MarineDebrisSAM-Net by <strong>Lakshay Jain</strong> (TIET Patiala) — dual-stream MobileNetV3-Small + SAM cross-attention on TrashCan-1.0.
        </p>

        <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
          <a className="pill-btn pill-dark" href={assetUrl('docs/MarineDebrisSAM_report.pdf')} target="_blank" rel="noreferrer">
            <ExternalLink size={14} /> Open Report PDF
          </a>
          <a className="pill-btn pill-outline" href={assetUrl('docs/research_poster.png')} target="_blank" rel="noreferrer">
            <ExternalLink size={14} /> View Poster
          </a>
        </div>
      </div>

      <div className="two-col-equal">
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <h2 className="section-heading" style={{ margin: 0 }}>Model Checkpoints</h2>
            <button className="pill-btn pill-outline" onClick={onRefreshHealth} style={{ padding: '0.32rem 0.7rem', fontSize: '0.75rem' }}>
              <RefreshCw size={12} /> Refresh
            </button>
          </div>
          <p className="section-sub">
            Drop your Colab <code>.pth</code> files into <code>checkpoints/</code> for live PyTorch inference.
          </p>
          {health ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
              {Object.entries(health.expected_checkpoints).map(([id, file]) => (
                <span key={id} className={`tag ${health.weights_loaded[id] ? 'tag-green' : 'tag-amber'} mono`}>
                  {file}: {health.weights_loaded[id] ? 'loaded' : 'missing'}
                </span>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: '0.83rem', color: 'var(--ink-muted)', margin: 0 }}>
              Backend not running locally (using browser preview mode).
            </p>
          )}
        </div>

        <div className="card">
          <h2 className="section-heading">Run & Deploy</h2>
          <pre className="code-box">{`# Backend (Python 3.10+)
pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8000

# Docker (HF Spaces / Render)
docker build -t marinedebrissam-net .
docker run -p 7860:7860 -v ./checkpoints:/app/checkpoints marinedebrissam-net`}</pre>
        </div>
      </div>
    </div>
  );
}
