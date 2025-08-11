import { NextResponse } from 'next/server';
import { sql } from '@vercel/postgres';

export async function GET() {
  await sql`CREATE TABLE IF NOT EXISTS subscribers (email TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT NOW())`;
  const { rows } = await sql`SELECT email FROM subscribers ORDER BY email`;
  return NextResponse.json({ subscribers: rows.map(r => r.email) });
}
