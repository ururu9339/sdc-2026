export const MAX_CV_BYTES = 10 * 1024 * 1024;

export async function validateCvPdf(file: File): Promise<string | null> {
  if (!/\.pdf$/i.test(file.name) || (file.type && file.type !== 'application/pdf')) {
    return 'Please choose a PDF file.';
  }
  if (file.size === 0) return 'The PDF is empty. Please choose another file.';
  if (file.size > MAX_CV_BYTES) return 'Please choose a PDF smaller than 10 MB.';
  try {
    const header = await file.slice(0, 1024).text();
    if (!header.includes('%PDF-')) return 'This file does not appear to be a PDF.';
  } catch {
    return 'The file could not be read. Please choose it again.';
  }
  return null;
}
