import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NIRNAY | Infrastructure Project Monitoring',
  description: 'National Infrastructure Risk & Nodal Action Intelligence — predictive monitoring and early intervention.',
}
export const viewport: Viewport = { colorScheme: 'light', themeColor: '#173a60' }
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" className="bg-background"><body>{children}{process.env.NODE_ENV === 'production' && <Analytics />}</body></html>
}
