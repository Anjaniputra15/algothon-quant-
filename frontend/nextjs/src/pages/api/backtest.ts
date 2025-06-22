import type { NextApiRequest, NextApiResponse } from 'next';
import { spawn } from 'child_process';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  const { strategy } = req.body;
  if (!strategy) {
    res.status(400).json({ error: 'Missing strategy name' });
    return;
  }

  // Example: python -m backend.cli_backtest --prices prices.txt --strategy $name
  const py = spawn('python', [
    '-m', 'backend.cli_backtest',
    '--prices', 'prices.txt',
    '--strategy', strategy,
  ]);

  let output = '';
  let error = '';

  py.stdout.on('data', (data) => {
    output += data.toString();
  });
  py.stderr.on('data', (data) => {
    error += data.toString();
  });
  py.on('close', (code) => {
    if (code !== 0) {
      res.status(500).json({ error: error || 'Python process failed' });
      return;
    }
    // TODO: parse output to extract P&L and drawdown arrays
    res.status(200).json({ output });
  });
} 