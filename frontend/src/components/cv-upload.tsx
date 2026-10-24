import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { FileText, Upload, X } from 'lucide-react';
import { useLanguage } from '@/lib/language';
import { validateCvPdf } from '@/lib/cv-file';

export function CvUpload({ file, onChange }: { file: File | null; onChange: (file: File | null) => void }) {
  const { t } = useLanguage();
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const request = useRef(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => () => { request.current += 1; }, []);

  async function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    const current = ++request.current;
    onChange(null);
    setError(null);
    setChecking(true);
    const problem = await validateCvPdf(selected);
    if (current !== request.current) return;
    setChecking(false);
    setError(problem);
    if (!problem) onChange(selected);
    else if (input.current) input.current.value = '';
  }

  function remove() {
    request.current += 1;
    setChecking(false);
    setError(null);
    onChange(null);
    if (input.current) input.current.value = '';
  }

  return (
    <section className="cv-upload" aria-labelledby="cv-upload-title" data-testid="section-cv-upload">
      <div className="cv-upload-heading">
        <Upload size={24} aria-hidden="true" />
        <h2 id="cv-upload-title">{t('Upload your CV')}</h2>
      </div>
      <p id="cv-upload-help">{t('Choose a CV PDF instead of filling in the form. PDF only, up to 10 MB.')}</p>
      <input ref={input} id="cv-pdf" type="file" accept=".pdf,application/pdf" onChange={selectFile}
        aria-describedby={`cv-upload-help${error ? ' cv-upload-error' : ''}`} aria-invalid={Boolean(error)}
        className="sr-only" data-testid="input-cv-pdf" />
      <label htmlFor="cv-pdf" className="cv-choose">{t(file ? 'Replace PDF' : 'Choose PDF')} <FileText size={18} aria-hidden="true" /></label>
      <div aria-live="polite" aria-busy={checking}>
        {checking && <p>{t('Checking your PDF…')}</p>}
        {error && <p id="cv-upload-error" className="profile-error" role="alert">{t(error)}</p>}
        {file && <div className="cv-selected" data-testid="text-selected-cv">
          <div><strong>{file.name}</strong><p>{t('PDF selected. The manual form is not required.')}</p></div>
          <button type="button" className="cv-remove" onClick={remove} aria-label={t('Remove PDF')} data-testid="button-remove-cv"><X size={20} /></button>
        </div>}
      </div>
    </section>
  );
}
