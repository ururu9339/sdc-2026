import { useEffect, useState } from 'react';

export function TypewriterText({ text }: { text: string }) {
  const [visible, setVisible] = useState(0);
  useEffect(() => {
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let timer: ReturnType<typeof setTimeout>;
    const start = () => {
      clearTimeout(timer);
      if (motion.matches) {
        setVisible(text.length);
        return;
      }
      let count = 0;
      setVisible(0);
      const type = () => {
        count += 1;
        setVisible(count);
        if (count < text.length) timer = setTimeout(type, 90);
        else timer = setTimeout(() => {
          count = 0;
          setVisible(0);
          timer = setTimeout(type, 90);
        }, 3500);
      };
      timer = setTimeout(type, 90);
    };
    start();
    motion.addEventListener('change', start);
    return () => {
      clearTimeout(timer);
      motion.removeEventListener('change', start);
    };
  }, [text]);
  return (
    <span className="typewriter-text">
      <span className="sr-only">{text}</span>
      <span className="typewriter-measure" aria-hidden="true">{text}</span>
      <span className="typewriter-output" aria-hidden="true">{text.slice(0, visible)}<span className="typewriter-cursor" /></span>
    </span>
  );
}
