import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'CondMat Digest Subscription',
  description: 'Subscribe to daily cond-mat.str-el arXiv digests.'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
