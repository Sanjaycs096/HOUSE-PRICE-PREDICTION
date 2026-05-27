from flask import Blueprint, current_app, jsonify
import os
import csv

market_bp = Blueprint('market_api', __name__)


@market_bp.route('/market/summary')
def market_summary():
    data_file = os.path.join(current_app.root_path, 'dataset', 'house_price_data.csv')
    if not os.path.exists(data_file):
        return jsonify({'ok': False, 'error': 'dataset not found'}), 404

    sums = {}
    counts = {}
    try:
        with open(data_file, 'r', encoding='utf-8') as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                loc = row.get('Location') or 'Unknown'
                try:
                    price = float(row.get('Price') or 0)
                except Exception:
                    price = 0
                sums[loc] = sums.get(loc, 0.0) + price
                counts[loc] = counts.get(loc, 0) + 1
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

    summary = []
    for loc in sums:
        summary.append({'location': loc, 'avg_price': sums[loc] / counts[loc], 'count': counts[loc]})

    summary.sort(key=lambda x: x['avg_price'], reverse=True)
    top = summary[:12]
    return jsonify({'ok': True, 'summary': top})
