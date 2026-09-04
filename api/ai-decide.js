// Vercel serverless function.
// Receives a prompt from the frontend, attaches the API key, and forwards it
// to the Claude API. The key lives only in the Vercel project's environment
// variables (ANTHROPIC_API_KEY) and never reaches the browser.

export default async function handler(req, res) {
  // Allow cross-origin calls from the frontend
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(404).json({ error: 'Not found' });
    return;
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: 'ANTHROPIC_API_KEY not configured on server' });
    return;
  }

  try {
    const { prompt } = req.body;
    const claudeRes = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 400,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    if (!claudeRes.ok) {
      const errText = await claudeRes.text();
      throw new Error(`Claude API error ${claudeRes.status}: ${errText}`);
    }

    const data = await claudeRes.json();
    res.status(200).json({ text: data.content[0].text });
  } catch (err) {
    console.error('AI request failed:', err.message);
    res.status(500).json({ error: err.message });
  }
}
