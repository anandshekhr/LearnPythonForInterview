import { Component, OnInit, inject, signal } from '@angular/core';
import { ChartConfiguration } from 'chart.js';
import { forkJoin } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Overview } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { ChartComponent, EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

interface Series { weeks: string[]; jobs_discovered: number[]; applications_submitted: number[]; interviews: number[]; }
interface Breakdown {
  status_funnel: { status: string; label: string; count: number }[]; match_categories: Record<string, number>;
  score_distribution: { bucket: string; count: number }[]; top_missing_skills: { skill: string; jobs: number }[];
  applications_by_source: { source_type: string; total: number; submitted: number }[];
}

const css = (name: string, fallback: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, ChartComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Analytics & reports" subtitle="Computed from your own data.">
    <mat-form-field appearance="outline" class="compact"><mat-label>Period</mat-label>
      <mat-select [value]="weeks()" (selectionChange)="weeks.set($event.value); load()">
        <mat-option [value]="4">4 weeks</mat-option><mat-option [value]="12">12 weeks</mat-option><mat-option [value]="26">26 weeks</mat-option></mat-select></mat-form-field>
    <button mat-stroked-button (click)="api.download('analytics/applications.csv', 'applications.csv')"><mat-icon>download</mat-icon> Applications CSV</button>
  </jp-page-header>
  <jp-loading [loading]="loading()" />
  @if (o(); as o) {
    <section class="metric-grid small">
      <div class="metric"><div class="value">{{ o.applications_submitted }}</div><div class="label">Submitted</div></div>
      <div class="metric"><div class="value">{{ o.conversion.response_rate ?? '—' }}{{ o.conversion.response_rate !== null ? '%' : '' }}</div><div class="label">Response rate</div></div>
      <div class="metric"><div class="value">{{ o.conversion.interview_rate ?? '—' }}{{ o.conversion.interview_rate !== null ? '%' : '' }}</div><div class="label">Interview rate</div></div>
      <div class="metric"><div class="value">{{ o.conversion.offer_rate ?? '—' }}{{ o.conversion.offer_rate !== null ? '%' : '' }}</div><div class="label">Offer rate</div></div>
    </section>
  }
  @if (activity(); as cfg) {
    <div class="grid-2">
      <mat-card class="item-card"><h3>Weekly activity</h3><jp-chart [config]="cfg" label="Weekly jobs discovered, applications and interviews" /></mat-card>
      <mat-card class="item-card"><h3>Application funnel</h3><jp-chart [config]="funnel()!" label="Applications by status" /></mat-card>
      <mat-card class="item-card"><h3>Match score distribution</h3><jp-chart [config]="scores()!" label="Active jobs by match score" /></mat-card>
      <mat-card class="item-card"><h3>Most requested skills you lack</h3>
        @for (s of breakdown()?.top_missing_skills ?? []; track s.skill) {
          <div class="row"><span class="grow">{{ s.skill }}</span><span class="muted">{{ s.jobs }} {{ s.jobs === 1 ? 'job' : 'jobs' }}</span></div>
        } @empty { <p class="muted">No gaps found among relevant jobs.</p> }
        <p class="muted small">Consider these only if they fit your goals — matching never assumes skills you have not verified.</p>
      </mat-card>
    </div>
  } @else if (!loading()) { <jp-empty icon="insights" title="No data yet" text="Analytics appear once jobs are discovered." /> }`,
})
export class AnalyticsComponent implements OnInit {
  api = inject(ApiService);
  private notify = inject(NotifyService);
  loading = signal(true);
  weeks = signal(12);
  o = signal<Overview | null>(null);
  breakdown = signal<Breakdown | null>(null);
  activity = signal<ChartConfiguration | null>(null);
  funnel = signal<ChartConfiguration | null>(null);
  scores = signal<ChartConfiguration | null>(null);

  ngOnInit(): void { this.load(); }

  load(): void {
    this.loading.set(true);
    forkJoin({
      o: this.api.get<Overview>('analytics/overview/'), t: this.api.get<Series>('analytics/timeseries/', { weeks: this.weeks() }),
      b: this.api.get<Breakdown>('analytics/breakdowns/'),
    }).subscribe({
      next: ({ o, t, b }) => {
        this.o.set(o); this.breakdown.set(b); this.loading.set(false);
        const c1 = css('--chart-1', '#3b6fd8'), c2 = css('--chart-2', '#1f9d76'), c3 = css('--chart-3', '#d9822b');
        const grid = css('--border', '#ddd'), text = css('--text-muted', '#666');
        const axes = { x: { grid: { display: false }, ticks: { color: text } }, y: { beginAtZero: true, grid: { color: grid }, ticks: { color: text, precision: 0 } } };
        const legend = { labels: { color: text, boxWidth: 12 } };
        this.activity.set(t.weeks.length ? { type: 'line', data: { labels: t.weeks, datasets: [
          { label: 'Jobs discovered', data: t.jobs_discovered, borderColor: c1, backgroundColor: c1, tension: 0.3 },
          { label: 'Applications submitted', data: t.applications_submitted, borderColor: c2, backgroundColor: c2, tension: 0.3 },
          { label: 'Interviews', data: t.interviews, borderColor: c3, backgroundColor: c3, tension: 0.3 }] },
          options: { maintainAspectRatio: false, scales: axes, plugins: { legend } } } : null);
        const f = b.status_funnel.filter(s => s.count > 0);
        this.funnel.set({ type: 'bar', data: { labels: f.map(s => s.label), datasets: [{ label: 'Applications', data: f.map(s => s.count), backgroundColor: c1, borderRadius: 4 }] },
          options: { indexAxis: 'y', maintainAspectRatio: false, scales: { x: axes.y, y: { ...axes.x } }, plugins: { legend: { display: false } } } });
        this.scores.set({ type: 'bar', data: { labels: b.score_distribution.map(s => s.bucket), datasets: [{ label: 'Jobs', data: b.score_distribution.map(s => s.count), backgroundColor: c2, borderRadius: 4 }] },
          options: { maintainAspectRatio: false, scales: axes, plugins: { legend: { display: false } } } });
        if (!t.weeks.length && !o.jobs_discovered) this.activity.set(null);
      },
      error: e => { this.loading.set(false); this.notify.error(e); },
    });
  }
}
