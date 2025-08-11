import { NextRequest, NextResponse } from 'next/server';
import { sql } from '@vercel/postgres';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const email = (body?.email || '').toString().trim().toLowerCase();
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ message: 'Invalid email' }, { status: 400 });
  }
  await sql`CREATE TABLE IF NOT EXISTS subscribers (email TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT NOW())`; 
  try {
    await sql`INSERT INTO subscribers (email) VALUES (${email}) ON CONFLICT (email) DO NOTHING`;
    return NextResponse.json({ message: 'Subscribed' }, { status: 200 });
  } catch (e: any) {
    return NextResponse.json({ message: e?.message || 'Error' }, { status: 500 });
  }
}
