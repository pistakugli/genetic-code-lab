import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

const SPAMBASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data';
const OUT_DIR = path.join(__dirname, '../../audit');
const OUT_FILE = path.join(OUT_DIR, 'spambase.csv');

function downloadFile(url: string, dest: string): Promise<void> {
    return new Promise((resolve, reject) => {
        if (!fs.existsSync(OUT_DIR)) {
            fs.mkdirSync(OUT_DIR, { recursive: true });
        }

        const file = fs.createWriteStream(dest);
        https.get(url, (response) => {
            if (response.statusCode === 200) {
                response.pipe(file);
                file.on('finish', () => {
                    file.close();
                    resolve();
                });
            } else {
                reject(new Error(`Failed to download: ${response.statusCode}`));
            }
        }).on('error', (err) => {
            fs.unlink(dest, () => reject(err));
        });
    });
}

async function main() {
    console.log(`Downloading Spambase dataset from UCI...`);
    try {
        await downloadFile(SPAMBASE_URL, OUT_FILE);
        
        // Count rows to verify
        const data = fs.readFileSync(OUT_FILE, 'utf8');
        const rows = data.split('\n').filter(l => l.trim() !== '');
        
        console.log(`✅ Successfully downloaded Spambase.`);
        console.log(`   Total records: ${rows.length}`);
        if (rows.length > 0) {
            const cols = rows[0].split(',');
            console.log(`   Total columns: ${cols.length} (57 features + 1 label)`);
        }
    } catch (e) {
        console.error('Error downloading:', e);
    }
}

main();
