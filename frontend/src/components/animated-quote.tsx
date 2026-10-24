import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { useLanguage } from '@/lib/language';

const outline = 'M211 249 H79 Q49 249 49 219 V42 H218 Q257 42 257 81 V202 C237 201 223 187 209 173 L155 147';
const loop = 'M264 252 L204 210 C189 198 177 207 157 207 C119 207 89 180 89 144 C89 110 117 83 151 83 C185 83 213 110 213 145 V183';
type Glyph = { letter: string; x: number; y: number; size: number; outlined: boolean; targetX: number; targetY: number };

export function AnimatedQuote() {
  const { language, t } = useLanguage();
  const paths = useRef<SVGGElement>(null);
  const [letters, setLetters] = useState<Glyph[]>([]);
  const [brandLetters, setBrandLetters] = useState<Glyph[]>([]);

  useEffect(() => {
    let active = true;
    const layout = () => {
      if (!active) return;
      const context = document.createElement('canvas').getContext('2d');
      if (!context || !paths.current) return;
      const family = getComputedStyle(document.body).fontFamily;
      const rows = language === 'ms'
        ? ['KERJA', 'BERUBAH.', 'ANDA', 'JUGA', 'BOLEH.']
        : ['WORK', 'IS', 'CHANGING.', 'SO CAN', 'YOU.'];
      const glyphs = rows.flatMap((row, rowIndex) => {
        let size = 112;
        context.font = `700 ${size}px ${family}`;
        const baseWidth = [...row].reduce((sum, letter) => sum + context.measureText(letter).width, 0) - (row.length - 1) * size * .045;
        if (baseWidth > 600) size *= 600 / baseWidth;
        context.font = `700 ${size}px ${family}`;
        const advances = [...row].map((letter) => context.measureText(letter).width);
        const spacing = size * .045;
        const width = advances.reduce((sum, advance) => sum + advance, 0) - (row.length - 1) * spacing;
        let x = (640 - width) / 2;
        return [...row].map((letter, index) => {
          const glyph = { letter, x, y: 98 + rowIndex * 105, size, outlined: rowIndex === (language === 'ms' ? 1 : 2), targetX: 0, targetY: 0 };
          x += advances[index] - spacing;
          return glyph;
        });
      });
      const strokes = Array.from(paths.current.querySelectorAll('path'));
      const lengths = strokes.map((path) => path.getTotalLength());
      const total = lengths.reduce((sum, length) => sum + length, 0);
      const projectToSymbol = (source: Glyph[]) => source.map((glyph, index) => {
        let distance = total * index / source.length;
        let pathIndex = 0;
        while (pathIndex < lengths.length - 1 && distance > lengths[pathIndex]) {
          distance -= lengths[pathIndex++];
        }
        const point = strokes[pathIndex].getPointAtLength(distance);
        return { ...glyph, targetX: 46 + point.x * 1.75, targetY: 20 + point.y * 1.75 };
      });
      setLetters(projectToSymbol(glyphs));
      const brand = 'currvix.ai';
      const brandSize = 92;
      context.font = `700 ${brandSize}px ${family}`;
      const brandWidth = context.measureText(brand).width;
      let brandX = (640 - brandWidth) / 2;
      setBrandLetters(projectToSymbol([...brand].map((letter) => {
        const glyph = { letter, x: brandX, y: 300, size: brandSize, outlined: false, targetX: 0, targetY: 0 };
        brandX += context.measureText(letter).width;
        return glyph;
      })));
    };
    layout();
    void document.fonts.ready.then(layout);
    document.fonts.addEventListener('loadingdone', layout);
    return () => {
      active = false;
      document.fonts.removeEventListener('loadingdone', layout);
    };
  }, [language]);

  return (
    <div className="quote-animation" data-testid="animated-quote">
      <h1 id="hero-title" className="sr-only">{t('Work is changing. So can you.')} currvix.ai</h1>
      <svg key={language} className="quote-stage" viewBox="0 0 640 555" aria-hidden="true">
        {letters.map((glyph, index) => (
          <g key={index} transform={`translate(${glyph.x} ${glyph.y})`}>
            <text
              className={`quote-letter${glyph.outlined ? ' quote-outline' : ''}`}
              fontSize={glyph.size}
              style={{
                '--move-x': `${320 - glyph.x}px`,
                '--move-y': `${280 - glyph.y}px`,
                '--turn': `${index % 2 ? -30 : 30}deg`,
              } as CSSProperties}
            >{glyph.letter}</text>
          </g>
        ))}
        {brandLetters.map((glyph, index) => (
          <g key={`brand-${index}`} transform={`translate(${glyph.x} ${glyph.y})`}>
            <text className="quote-brand-letter" fontSize={glyph.size} style={{
              '--move-x': `${glyph.targetX - glyph.x}px`,
              '--move-y': `${glyph.targetY - glyph.y}px`,
              '--turn': `${index % 2 ? -20 : 20}deg`,
            } as CSSProperties}>{glyph.letter}</text>
          </g>
        ))}
        <g ref={paths} className="quote-symbol" transform="translate(46 20) scale(1.75)" fill="none" stroke="currentColor" strokeWidth="20" strokeLinejoin="round">
          <path d={outline} pathLength="1" />
          <path d={loop} pathLength="1" />
        </g>
      </svg>
    </div>
  );
}
