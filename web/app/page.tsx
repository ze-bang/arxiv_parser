"use client";
import { useState } from 'react';

export default function Home() {
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState<string | null>(null);

  const subscribe = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus(null);
    const res = await fetch('/api/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    const j = await res.json();
    setStatus(j.message || (res.ok ? 'Subscribed!' : 'Failed'));
  };

  return (
    <div className="container">
      <h1>CondMat Digest Subscription</h1>
      <p>Enter your email to receive the top daily cond-mat.str-el papers.</p>
      <form className="card" onSubmit={subscribe}>
        <label htmlFor="email">Email</label>
        <input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} required style={{width:'100%'}} />
        <div style={{marginTop:12}}>
          <button type="submit">Subscribe</button>
        </div>
      </form>
      {status && <p style={{marginTop:12}}>{status}</p>}
    </div>
  );
}
