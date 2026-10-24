import { type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { ArrowRight, ArrowUpRight } from 'lucide-react';
import { AnimatedQuote } from '@/components/animated-quote';
import { TypewriterText } from '@/components/typewriter-text';
import { ProfileForm } from '@/components/profile-form';
import { MarketSignals } from '@/components/market-signals';
import { LanguageProvider, useLanguage } from '@/lib/language';
import {
  Link,
  Route,
  Switch,
  useLocation,
  Router as WouterRouter,
} from 'wouter';

const queryClient = new QueryClient();

function Brand() {
  const { t } = useLanguage();
  return <Link href="/" className="brand" aria-label={t('currvix.ai home')}>currvix.ai</Link>;
}

function MainNav({ explore = false }: { explore?: boolean }) {
  const { language, setLanguage, t } = useLanguage();
  return (
    <header className={explore ? 'explore-nav' : 'nav'}>
      <Brand />
      <nav className="navlinks" aria-label={t('Main navigation')}>
        <select className="language-switch" aria-label={t('Language')} value={language} onChange={(event) => setLanguage(event.target.value === 'ms' ? 'ms' : 'en')} data-testid="select-language">
          <option value="en">English</option>
          <option value="ms">Bahasa Melayu</option>
        </select>
        <Link href="/explore" aria-current={explore ? 'page' : undefined}>{t('Explore roles')} <ArrowUpRight size={13} strokeWidth={1.8} /></Link>
      </nav>
    </header>
  );
}

function Home() {
  const { t } = useLanguage();
  return (
    <main className="landing page-enter">
      <div className="landing-grid" aria-hidden="true" />
      <MainNav />
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-quote">
          <AnimatedQuote />
        </div>
        <div className="hero-copy">
          <p>{t('Career choices shouldn’t feel like guesswork. Currvix makes the signals in a changing job market easier to see — and the next skill easier to name.')}</p>
          <Link className="cta" href="/explore" data-testid="link-explore-signal-map">{t('Explore the signal map')} <span><ArrowRight size={23} /></span></Link>
          <section className="how-to" aria-labelledby="how-to-heading" data-testid="section-getting-started">
            <h2 id="how-to-heading"><TypewriterText text={t('How to get started')} /></h2>
            <ol>
              <li>{t('Click on Explore the signal map')}</li>
              <li>{t('Fill up the form or choose a CV PDF')}</li>
              <li>{t('Click Submit profile')}</li>
            </ol>
            <p className="submit-notice" role="note" aria-label={t('Submission note')}>{t('The Submit profile button appears after you select a valid PDF or complete all five fields.')}</p>
          </section>
        </div>
        <div className="hero-note">{t('For workers · students · career counsellors')}</div>
      </section>
    </main>
  );
}

function Explore() {
  const { t } = useLanguage();
  return (
    <main className="explore-page page-enter">
      <MainNav explore />
      <div className="explore-wrap">
        <div className="explore-topline"><span>{t('Currvix / Signal map')}</span><span className="live-dot">{t('Prototype · unvalidated forecasts')}</span></div>
        <div className="explore-title">
          <h1>{t('Find the work taking shape.')}</h1>
        </div>
        <ProfileForm />
        <MarketSignals />
      </div>
    </main>
  );
}

function Router() {
  return (
    // Keep a shared shell (sidebar, navbar) outside the boundary so it
    // survives a page crash.
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/explore" component={Explore} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
      </LanguageProvider>
    </QueryClientProvider>
  );
}

export default App;
