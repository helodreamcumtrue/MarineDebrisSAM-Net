import { ArrowRight, Cpu, Eye, Layers, Sparkles } from 'lucide-react';

interface HomeTabProps {
  onGoDemo: () => void;
  onGoResults: () => void;
  onGoReport: () => void;
}

const ARCH_CARDS = [
  {
    num: '01',
    bg: 'var(--sage-soft)',
    icon: Eye,
    title: 'A1 · Original Only',
    desc: '88.10% accuracy on raw underwater crops.',
  },
  {
    num: '02',
    bg: 'var(--alabaster)',
    icon: Layers,
    title: 'A2 · SAM Only',
    desc: '70.31% accuracy using isolated object masks.',
  },
  {
    num: '03',
    bg: 'var(--mist-blue)',
    icon: Cpu,
    title: 'A3 · Concat Fusion',
    desc: '88.96% top accuracy combining both streams.',
  },
  {
    num: '04',
    bg: 'var(--warm-sand)',
    icon: Sparkles,
    title: 'A5 · MDMAM Attention',
    desc: '0.7176 Macro F1 with Q-K-V cross-attention.',
  },
];

export function HomeTab({ onGoDemo, onGoResults, onGoReport }: HomeTabProps) {
  return (
    <div>
      {/* Clean Hero */}
      <section className="hero-grid" style={{ marginBottom: '2rem' }}>
        <div>
          <div className="kicker">TrashCan-1.0 · 6 Debris Classes</div>
          <h1 className="display-h1">
            Classify marine debris
            <br />
            <span className="muted">with dual-stream clarity.</span>
          </h1>
          <p style={{ fontSize: '0.94rem', color: 'var(--ink-muted)', maxWidth: '430px', margin: '0 0 1.4rem' }}>
            Fuses raw underwater RGB imagery with SAM foreground masks using lightweight MobileNetV3 backbones and cross-attention.
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.65rem' }}>
            <button className="pill-btn pill-dark" onClick={onGoDemo}>
              Open Live Demo <ArrowRight size={14} />
            </button>
            <button className="pill-btn pill-sage" onClick={onGoResults}>
              View Results
            </button>
          </div>
        </div>

        <div className="hero-visual" style={{ aspectRatio: '4 / 3' }}>
          <img
            src="/samples/sample_1_metal_orig.png"
            alt="Underwater ROV grasping metallic marine debris"
          />
        </div>
      </section>

      {/* Essential Metrics Strip */}
      <section className="metrics-strip" style={{ marginBottom: '2rem' }}>
        <div>
          <div className="metric-num">4,662</div>
          <div className="metric-desc">Paired RGB + SAM images</div>
        </div>
        <div>
          <div className="metric-num">88.96%</div>
          <div className="metric-desc">Best raw accuracy (A3)</div>
        </div>
        <div>
          <div className="metric-num">0.7176</div>
          <div className="metric-desc">Best Macro F1 (MDMAM)</div>
        </div>
        <div
          onClick={onGoReport}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && onGoReport()}
          style={{
            background: 'var(--alabaster)',
            borderRadius: '14px',
            padding: '0.8rem 1rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
            border: '1px solid var(--border)',
          }}
        >
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--ink-subtle)' }}>Documentation</div>
            <div style={{ fontSize: '0.84rem', fontWeight: 600 }}>Report PDF & Setup</div>
          </div>
          <span
            style={{
              width: '30px',
              height: '30px',
              borderRadius: '50%',
              background: 'var(--sage-soft)',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            <ArrowRight size={14} />
          </span>
        </div>
      </section>

      {/* 4 Compact Pastel Cards */}
      <section className="pastel-grid" style={{ marginBottom: 0 }}>
        {ARCH_CARDS.map((c) => {
          const Icon = c.icon;
          return (
            <div
              key={c.num}
              className="pastel-card"
              onClick={onGoResults}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && onGoResults()}
              style={{ background: c.bg, minHeight: '165px', cursor: 'pointer' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <span
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '9px',
                    background: 'rgba(255,255,255,0.85)',
                    display: 'grid',
                    placeItems: 'center',
                  }}
                >
                  <Icon size={15} />
                </span>
                <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--ink-subtle)' }}>
                  {c.num}
                </span>
              </div>
              <div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '0.25rem' }}>{c.title}</div>
                <p style={{ fontSize: '0.79rem', color: 'var(--ink-muted)', margin: 0 }}>{c.desc}</p>
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
