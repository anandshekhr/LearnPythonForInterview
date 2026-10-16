import { Component, OnInit, inject, signal } from '@angular/core';
import { PageEvent } from '@angular/material/paginator';
import { ApiService } from '../core/api.service';
import { JobMatch, Paginated } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent, ScoreBadgeComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, ScoreBadgeComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Recommended jobs" subtitle="Explainable match scores based on your verified profile and preferences.">
    <button mat-stroked-button (click)="rematch()"><mat-icon>refresh</mat-icon> Recompute scores</button>
  </jp-page-header>
  <div class="toolbar-row wrap">
    <mat-button-toggle-group [value]="category()" (change)="setCategory($event.value)" aria-label="Category">
      <mat-button-toggle value="">All</mat-button-toggle><mat-button-toggle value="excellent">Excellent</mat-button-toggle>
      <mat-button-toggle value="good">Good</mat-button-toggle><mat-button-toggle value="possible">Possible</mat-button-toggle>
      <mat-button-toggle value="low">Low</mat-button-toggle>
    </mat-button-toggle-group>
    <mat-form-field appearance="outline"><mat-label>Show</mat-label>
      <mat-select [value]="state()" (selectionChange)="setState($event.value)">
        <mat-option value="">Not dismissed</mat-option><mat-option value="shortlisted">Shortlisted</mat-option>
        <mat-option value="new">New</mat-option><mat-option value="dismissed">Dismissed</mat-option></mat-select></mat-form-field>
  </div>
  <jp-loading [loading]="loading()" />
  @for (m of matches(); track m.id) {
    <mat-card class="match-card">
      <div class="row">
        <jp-score [score]="m.score" [category]="m.category" />
        <a class="grow" [routerLink]="['/jobs', m.job.id]">
          <div class="strong">{{ m.job.title }}</div>
          <div class="muted">{{ m.job.company_name }} · {{ m.job.location_raw || m.job.work_model }} @if (m.user_state === 'new') {<span class="badge">new</span>}</div>
        </a>
        <button mat-icon-button [matTooltip]="m.user_state === 'shortlisted' ? 'Shortlisted' : 'Shortlist'" (click)="setUserState(m, m.user_state === 'shortlisted' ? 'viewed' : 'shortlisted')">
          <mat-icon>{{ m.user_state === 'shortlisted' ? 'bookmark' : 'bookmark_border' }}</mat-icon></button>
        <button mat-icon-button matTooltip="Dismiss" (click)="setUserState(m, 'dismissed')"><mat-icon>visibility_off</mat-icon></button>
      </div>
      <p class="explanation">{{ m.explanation }}</p>
      <div class="chips">
        @for (s of m.skills_matched.slice(0, 8); track s) { <span class="chip ok">{{ s }}</span> }
        @for (s of m.skills_missing.slice(0, 5); track s) { <span class="chip missing">{{ s }}</span> }
      </div>
      <div class="muted small">{{ m.recommended_action }}</div>
    </mat-card>
  } @empty {
    @if (!loading()) { <jp-empty icon="recommend" title="No recommendations" text="Verify your profile and add job sources; matches appear after the next sync." /> }
  }
  <mat-paginator [length]="count()" [pageSize]="20" [pageIndex]="page() - 1" (page)="pageChange($event)" />`,
})
export class RecommendedComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  loading = signal(true);
  matches = signal<JobMatch[]>([]);
  count = signal(0);
  page = signal(1);
  category = signal('');
  state = signal('');

  ngOnInit(): void { this.load(); }

  load(): void {
    this.loading.set(true);
    this.api.get<Paginated<JobMatch>>('matches/', {
      category: this.category(), user_state: this.state(), job_status: 'active', hide_excluded: true, ordering: '-score', page: this.page(),
    }).subscribe({
      next: r => {
        const rows = this.state() ? r.results : r.results.filter(m => m.user_state !== 'dismissed');
        this.matches.set(rows); this.count.set(r.count); this.loading.set(false);
      },
      error: e => { this.loading.set(false); this.notify.error(e); },
    });
  }
  setCategory(v: string): void { this.category.set(v); this.page.set(1); this.load(); }
  setState(v: string): void { this.state.set(v); this.page.set(1); this.load(); }
  pageChange(e: PageEvent): void { this.page.set(e.pageIndex + 1); this.load(); }

  setUserState(m: JobMatch, user_state: string): void {
    this.api.patch<JobMatch>(`matches/${m.id}/`, { user_state }).subscribe({
      next: u => this.matches.update(l => user_state === 'dismissed' && !this.state() ? l.filter(x => x.id !== m.id) : l.map(x => x.id === u.id ? u : x)),
      error: e => this.notify.error(e),
    });
  }
  rematch(): void {
    this.api.post('matches/rematch/').subscribe({ next: () => { this.notify.success('Scores recomputed.'); this.load(); }, error: e => this.notify.error(e) });
  }
}
