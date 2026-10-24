import { RotateCcw } from 'lucide-react';
import type { Outlook, Report } from '@/lib/api';
import { useLanguage } from '@/lib/language';
import { outlookChip, percent, readableId } from '@/lib/outlook';

export function OutlookChip({ outlook, validated, window, stale }: { outlook: Outlook; validated: boolean; window?: string | null; stale?: boolean | null }) {
  const { t } = useLanguage();
  const chip = outlookChip(outlook, validated);
  return (
    <span className="outlook">
      <span className={`outlook-chip outlook-${chip.tone}`}>
        {t(chip.label)}{chip.unconfirmed ? `, ${t('unconfirmed')}` : ''}
        {chip.tone !== 'none' && outlook.growth !== null ? ` · ${percent(outlook.growth)}` : ''}
      </span>
      {chip.tone !== 'none' && window && <span className="outlook-window">{window}</span>}
      {chip.tone !== 'none' && stale && <span className="outlook-stale">{t('out of date')}</span>}
      {outlook.shares_signal_with && <span className="outlook-window">{t('same signal as')} {readableId(outlook.shares_signal_with)}</span>}
    </span>
  );
}

export function ReportView({ report, onEdit }: { report: Report; onEdit: () => void }) {
  const { t } = useLanguage();
  const recognised = report.resolved_skills.filter((s) => s.skill_id && s.status !== 'unresolved');
  const names = [...new Map(recognised.map((s) => [s.skill_id, s.name ?? s.skill_id])).values()];
  const gapById = new Map(report.skill_gaps.map((g) => [g.skill_id, g]));
  const matchedBy = report.engine === 'qwen_pgvector' ? t('Matched with Qwen3 embeddings') : t('Matched with the offline skill dictionary');

  return (
    <section className="report" aria-live="polite" data-testid="section-report">
      {report.banner && <p className="report-banner" role="status" data-testid="text-report-banner">{report.banner}</p>}

      <div className="detail-eyebrow"><span>{t('Your report')}</span><span>{matchedBy}</span></div>
      {report.message && <p className="report-message" data-testid="text-report-message">{report.message}</p>}

      <div className="report-block">
        <h3>{t('Skills we recognised')}</h3>
        {names.length ? <ul className="chip-list">{names.map((name) => <li key={name}>{name}</li>)}</ul> : <p>{t('None yet. Try listing your skills separated by commas.')}</p>}
        {report.unresolved.length > 0 && <p className="report-muted">{t('Not recognised')}: {report.unresolved.join(', ')}</p>}
      </div>

      {report.best_fit_jobs.length > 0 && (
        <div className="report-block">
          <h3>{t('Jobs closest to your profile')}</h3>
          <ol className="job-list">
            {report.best_fit_jobs.map((job) => (
              <li key={job.job_id} className="job-card" data-testid={`card-job-${job.job_id}`}>
                <div className="job-head">
                  <strong>{job.name}</strong>
                  <span className="job-coverage">{Math.round(job.coverage * 100)}% {t('overlap')}</span>
                </div>
                <div className="coverage-bar" aria-hidden="true"><span style={{ width: `${Math.min(100, Math.round(job.coverage * 100))}%` }} /></div>
                <OutlookChip outlook={job} validated={report.validated} window={report.forecast_window} stale={report.stale} />
                {job.missing.length > 0 && <p className="report-muted">{t('Missing')}: {job.missing.map((m) => m.name).join(', ')}</p>}
              </li>
            ))}
          </ol>
        </div>
      )}

      {report.skill_gaps.length > 0 && (
        <div className="report-block">
          <h3>{t('Your reskilling path')}</h3>
          <p className="report-muted">{report.learning_path.note}</p>
          <ol className="path-list">
            {report.learning_path.stages.filter((stage) => stage.skills.length).map((stage) => (
              <li key={stage.title} className="path-stage">
                <span className="path-title">{t(stage.title)}</span>
                <ul>
                  {stage.skills.map((skill) => {
                    const gap = gapById.get(skill.skill_id);
                    return (
                      <li key={skill.skill_id} data-testid={`gap-${skill.skill_id}`}>
                        <strong>{skill.name}</strong>
                        {gap && <OutlookChip outlook={gap} validated={report.validated} stale={report.stale} />}
                        {gap && <p className="report-muted">{gap.why}</p>}
                      </li>
                    );
                  })}
                </ul>
              </li>
            ))}
          </ol>
        </div>
      )}

      {report.market_snapshot.length > 0 && (
        <div className="report-block">
          <h3>{t('Past share of postings')}</h3>
          <p className="report-muted">{t('Descriptive history from our data, not a forecast.')}</p>
          <dl className="snapshot-list">
            {report.market_snapshot.map((item) => (
              <div key={item.entity_id}><dt>{item.name}</dt><dd>{item.percent_change === null ? '—' : percent(item.percent_change / 100, 0)}</dd></div>
            ))}
          </dl>
        </div>
      )}

      {report.warnings.length > 0 && (
        <ul className="report-notes">{report.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
      )}
      <ul className="report-notes">{report.caveats.map((c) => <li key={c}>{c}</li>)}</ul>

      <button type="button" className="profile-submit" onClick={onEdit} data-testid="button-edit-profile">
        {t('Edit details')} <RotateCcw size={16} aria-hidden="true" />
      </button>
    </section>
  );
}
