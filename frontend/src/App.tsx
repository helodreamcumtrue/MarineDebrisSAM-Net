import { useCallback, useEffect, useState } from 'react';
import { Compass } from 'lucide-react';
import { HealthStatus, fetchHealth } from './api/client';
import { AblationTab } from './components/AblationTab';
import { HomeTab } from './components/HomeTab';
import { InferenceTab } from './components/InferenceTab';
import { ReportTab } from './components/ReportTab';

type TabId = 'home' | 'demo' | 'results' | 'report';

export default function App() {
  const [tab, setTab] = useState<TabId>('home');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const refreshHealth = useCallback(async () => {
    const h = await fetchHealth();
    setHealth(h);
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  return (
    <div className="outer-wrap">
      <div className="page-sheet">
        <header className="navbar">
          <button className="nav-brand" onClick={() => setTab('home')}>
            <span className="nav-brand-badge">
              <Compass size={15} />
            </span>
            MarineSAM-Net
          </button>

          <nav className="nav-links" aria-label="Main navigation">
            <button
              className={`nav-link ${tab === 'home' ? 'active' : ''}`}
              onClick={() => setTab('home')}
            >
              Home
            </button>
            <button
              className={`nav-link ${tab === 'demo' ? 'active' : ''}`}
              onClick={() => setTab('demo')}
            >
              Live Demo
            </button>
            <button
              className={`nav-link ${tab === 'results' ? 'active' : ''}`}
              onClick={() => setTab('results')}
            >
              Results
            </button>
            <button
              className={`nav-link ${tab === 'report' ? 'active' : ''}`}
              onClick={() => setTab('report')}
            >
              Report
            </button>
          </nav>

          <div className="nav-right" />
        </header>

        <main className="main-body">
          {tab === 'home' && (
            <HomeTab
              onGoDemo={() => setTab('demo')}
              onGoResults={() => setTab('results')}
              onGoReport={() => setTab('report')}
            />
          )}
          {tab === 'demo' && <InferenceTab health={health} />}
          {tab === 'results' && <AblationTab />}
          {tab === 'report' && <ReportTab health={health} onRefreshHealth={refreshHealth} />}
        </main>

        <footer className="sheet-footer">
          <span>MarineDebrisSAM-Net · Lakshay Jain (TIET Patiala)</span>
          <span>TrashCan-1.0 · PyTorch</span>
        </footer>
      </div>
    </div>
  );
}
