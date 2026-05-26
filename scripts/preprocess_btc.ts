import * as fs from 'fs';

function preprocessBtc() {
    const raw = fs.readFileSync('btc_raw.csv', 'utf-8');
    const lines = raw.split('\n');
    const prices: [number, number][] = [];

    // Skip header and empty lines
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const [dateStr, priceStr] = line.split(',');
        const timestamp = new Date(dateStr).getTime();
        const price = parseFloat(priceStr);
        
        if (!isNaN(timestamp) && !isNaN(price)) {
            prices.push([timestamp, price]);
        }
    }

    // Sort by timestamp just in case
    prices.sort((a, b) => a[0] - b[0]);

    fs.writeFileSync('btc_data.json', JSON.stringify({ prices }, null, 2));
    console.log(`Successfully processed ${prices.length} price points into btc_data.json`);
}

preprocessBtc();
