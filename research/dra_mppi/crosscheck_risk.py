"""Independent probability diagnostics, not navigation or a new estimator."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import ncx2
import torch
from run_upstream_audit import load_module, make_obstacles, save_csv


def quadrature(mu, q, sigma, radius):
    nodes, weights = np.polynomial.legendre.leggauss(80)
    radial = (nodes + 1) * radius / 2
    radial_w = weights * radius / 2
    angle = np.arange(256) * 2 * np.pi / 256
    points = q + np.stack((radial[:, None] * np.cos(angle), radial[:, None] * np.sin(angle)), -1)
    delta = points[:, :, None, :] - mu
    pdf = np.exp(-np.sum(delta ** 2, -1) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    area_weight = radial[:, None] * radial_w[:, None] * (2 * np.pi / 256)
    individual = np.sum(pdf * area_weight[:, :, None], axis=(0, 1))
    density_expression = np.sum((1 - np.prod(1 - pdf, axis=-1)) * area_weight)
    return 1 - np.prod(1 - individual), float(density_expression)


def direct_mc(mu, queries, sigma, radius, seed, n=200000):
    generator = np.random.default_rng(seed)
    counts = np.zeros(len(queries), dtype=np.int64)
    for start in range(0, n, 10000):
        size = min(10000, n - start)
        targets = generator.normal(size=(size, len(mu), 2)) * sigma + mu
        distance2 = np.sum((targets[:, None, :, :] - queries[None, :, None, :]) ** 2, -1)
        counts += np.any(distance2 <= radius ** 2, axis=-1).sum(axis=0)
    p = counts / n
    # Wilson 95% interval, including near-zero counts.
    z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return p, center-half, center+half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cls = load_module(args.upstream/'src/mppi_planner/src/prob_mppi/dynamic_obstacles.py', 'source_risk').DynamicObstacles
    torch.set_num_threads(2)
    layouts = {
        'two_separated_means': np.array([[-.6, 0.], [.6, 0.]]),
        'four_separated_means': np.array([[-1.2, 0.], [0., 1.2], [1.2, 0.], [0., -1.2]]),
        'overlapping_ring_stress': np.stack((.55*np.cos(np.arange(12)*2*np.pi/12), .55*np.sin(np.arange(12)*2*np.pi/12)), -1),
    }
    queries = np.stack((np.linspace(-.9, .9, 7), np.zeros(7)), -1)
    rows = []
    for name, means in layouts.items():
        for sigma in [.2, .6]:
            for radius in [.3, .825]:
                torch.manual_seed(17)
                obj, cfg = make_obstacles(cls, torch.tensor(means[None], dtype=torch.float32), sigma, len(queries), 20000)
                obj.integral_radius = radius
                cfg['obstacles']['integral_radius'] = radius
                x = torch.tensor(queries[:, 0:1], dtype=torch.float32)
                y = torch.tensor(queries[:, 1:2], dtype=torch.float32)
                upstream = obj.integrate_one_shot_monte_carlo_circles(x, y).numpy()[:, 0]
                cdf = ncx2.cdf(radius**2/sigma**2, 2, np.sum((queries[:, None]-means[None])**2, -1)/sigma**2)
                event_union = 1-np.prod(1-cdf, axis=-1)
                mc, lower, upper = direct_mc(means, queries, sigma, radius, 301)
                for j, q in enumerate(queries):
                    event_quad, density_quad = quadrature(means, q, sigma, radius)
                    if abs(event_quad-event_union[j]) > 1e-7:
                        raise AssertionError('Two deterministic event references disagree')
                    tolerance = max(.004, 6*np.sqrt(event_union[j]*(1-event_union[j])/200000))
                    if abs(mc[j]-event_union[j]) > tolerance:
                        raise AssertionError('Independent target-sampling reference disagrees')
                    rows.append({'layout': name, 'sigma_m': sigma, 'radius_m': radius,
                                 'query_x': q[0], 'query_y': q[1], 'upstream_value': float(upstream[j]),
                                 'independent_event_cdf': float(event_union[j]),
                                 'independent_event_quadrature': event_quad,
                                 'independent_target_mc': mc[j], 'mc_wilson_low': lower[j], 'mc_wilson_high': upper[j],
                                 'source_density_expression_quadrature': density_quad,
                                 'risk_threshold': .05,
                                 'upstream_accepts_reference_rejects': bool(upstream[j] <= .05 < event_union[j])})
    save_csv(args.output/'risk_crosscheck.csv', rows)
    result = {'scope': 'Constructed independent isotropic Gaussian inputs; no navigation episodes',
              'queries': len(rows), 'direct_target_samples_per_fixture': 200000,
              'max_mc_vs_event_cdf_abs_error': max(abs(r['independent_target_mc']-r['independent_event_cdf']) for r in rows),
              'max_quadrature_vs_cdf_abs_error': max(abs(r['independent_event_quadrature']-r['independent_event_cdf']) for r in rows),
              'max_source_vs_its_density_expression_abs_error': max(abs(r['upstream_value']-r['source_density_expression_quadrature']) for r in rows),
              'max_source_vs_event_union_abs_difference': max(abs(r['upstream_value']-r['independent_event_cdf']) for r in rows),
              'threshold_disagreement_queries': sum(r['upstream_accepts_reference_rejects'] for r in rows),
              'warning': 'References validate probability calculations for these inputs, not full-paper claims or a novel improvement.'}
    (args.output/'risk_crosscheck.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
