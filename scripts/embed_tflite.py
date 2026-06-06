#!/usr/bin/env python3
"""Embed a .tflite file as a C byte array."""

import argparse
import os


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  parser.add_argument('--symbol', default='g_hey_siri_model_data')
  parser.add_argument('--length_symbol', default='g_hey_siri_model_data_len')
  args = parser.parse_args()

  with open(args.input, 'rb') as f:
    data = f.read()

  lines = []
  for i in range(0, len(data), 12):
    chunk = data[i:i + 12]
    hexes = ', '.join(f'0x{b:02x}' for b in chunk)
    lines.append('  ' + hexes + ',')

  content = f"""// Auto-generated from {os.path.basename(args.input)}

#include <cstddef>

alignas(8) extern const unsigned char {args.symbol}[] = {{
{chr(10).join(lines)}
}};

extern const size_t {args.length_symbol} = {len(data)};
"""

  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
  with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

  print(f'Wrote {args.output} ({len(data)} bytes)')


if __name__ == '__main__':
  main()
