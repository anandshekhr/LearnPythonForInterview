import { Component, OnInit, inject, signal } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { forkJoin } from 'rxjs';
import { ApiService } from '../core/api.service';
import { AutomationRun, AutomationSettings, JobSource, Paginated, ScheduledSearch, ScoringConfig } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { FormDialogComponent } from '../shared/form-dialog.component';
import { MATERIAL } from '../shared/material';
import { ConfirmDialogComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent],
  template: `
  <jp-page-header title="Automation settings" subtitle="Quality over volume: you decide how far automation goes." />
  <jp-loading [loading]="!s()" />
  @if (s(); as s) {
    <mat-tab-group animationDuration="0ms">
      <mat-tab label="Application mode">
        <div class="mode-grid">
          @for (m of modes; track m.value) {
            <mat-card class="mode-card" [class.selected]="s.application_mode === m.value" (click)="setMode(m.value)" tabindex="0" (keydown.enter)="setMode(m.value)">
              <mat-icon>{{ m.icon }}</mat-icon><h3>{{ m.title }}</h3><p>{{ m.text }}</p>
            </mat-card>
          }
        </div>
        @if (s.application_mode === 'automated') {
          <mat-card class="callout" [class.warn]="!s.automated_submission_allowed"><mat-icon>gavel</mat-icon>
            <div>Consent given {{ s.automated_submission_consent_at | date: 'medium' }}. Automated submission only works for boards where the
              employer has issued an API key. <button mat-button color="warn" (click)="revoke()">Revoke consent</button></div></mat-card>
        }
        <div class="form-grid">
          <mat-form-field appearance="outline"><mat-label>Max automated submissions per day</mat-label>
            <input matInput type="number" min="0" max="25" [(ngModel)]="s.max_submissions_per_day"></mat-form-field>
          <mat-form-field appearance="outline"><mat-label>Auto-shortlist jobs scoring at least</mat-label>
            <input matInput type="number" min="0" max="100" [(ngModel)]="s.auto_shortlist_min_score"><mat-hint>Leave empty to disable</mat-hint></mat-form-field>
        </div>
        <button mat-flat-button color="primary" (click)="save()">Save</button>
      </mat-tab>
      <mat-tab label="Notifications">
        <div class="toggle-list">
          <mat-slide-toggle [(ngModel)]="s.notify_new_matches">New high-match jobs</mat-slide-toggle>
          <mat-form-field appearance="outline"><mat-label>High-match threshold</mat-label><input matInput type="number" min="0" max="100" [(ngModel)]="s.notify_min_score"></mat-form-field>
          <mat-slide-toggle [(ngModel)]="s.notify_awaiting_review">Resumes awaiting review</mat-slide-toggle>
          <mat-slide-toggle [(ngModel)]="s.notify_submissions">Successful and failed submissions</mat-slide-toggle>
          <mat-slide-toggle [(ngModel)]="s.notify_interviews">Upcoming interviews</mat-slide-toggle>
          <mat-slide-toggle [(ngModel)]="s.notify_follow_ups">Follow-up reminders</mat-slide-toggle>
          <mat-slide-toggle [(ngModel)]="s.notify_expired_jobs">Expired postings you track</mat-slide-toggle>
          <mat-slide-toggle [(ngModel)]="s.email_notifications" disabled>Email notifications (coming later)</mat-slide-toggle>
        </div>
        <button mat-flat-button color="primary" (click)="save()">Save</button>
      </mat-tab>
      <mat-tab label="Scoring">
        @if (scoring(); as c) {
          <p class="muted">Weights are relative. Signals without information (e.g. undisclosed salary) are skipped, never counted as zero.</p>
          <div class="form-grid">
            @for (w of weights; track w.key) {
              <div><label class="small">{{ w.label }}: {{ c[w.key] }}</label>
                <mat-slider min="0" max="50" step="1" discrete class="full"><input matSliderThumb [(ngModel)]="c[w.key]"></mat-slider></div>
            }
            <mat-form-field appearance="outline"><mat-label>Excellent ≥</mat-label><input matInput type="number" [(ngModel)]="c['excellent_threshold']"></mat-form-field>
            <mat-form-field appearance="outline"><mat-label>Good ≥</mat-label><input matInput type="number" [(ngModel)]="c['good_threshold']"></mat-form-field>
            <mat-form-field appearance="outline"><mat-label>Possible ≥</mat-label><input matInput type="number" [(ngModel)]="c['possible_threshold']"></mat-form-field>
          </div>
          <button mat-flat-button color="primary" (click)="saveScoring()">Save & recompute</button>
        }
      </mat-tab>
      <mat-tab label="Scheduled searches">
        <div class="toolbar-row"><span class="grow muted">Scheduled searches synchronise sources through Celery Beat; new jobs are scored automatically.</span>
          <button mat-flat-button color="primary" (click)="editSearch()"><mat-icon>add</mat-icon> New schedule</button></div>
        @for (q of searches(); track q.id) {
          <mat-card class="item-card"><div class="row wrap">
            <mat-slide-toggle [checked]="q.enabled" (change)="patchSearch(q, { enabled: $event.checked })"></mat-slide-toggle>
            <div class="grow"><div class="strong">{{ q.name }}</div>
              <div class="muted small">every {{ q.interval_hours }} h · {{ q.sources.length ? q.sources.length + ' sources' : 'all enabled sources' }}
                · next {{ q.next_run_at | date: 'short' }} · last {{ q.last_run_at ? (q.last_run_at | date: 'short') : 'never' }}</div></div>
            <button mat-stroked-button (click)="runNow(q)"><mat-icon>play_arrow</mat-icon> Run now</button>
            <button mat-icon-button (click)="editSearch(q)" aria-label="Edit"><mat-icon>edit</mat-icon></button>
            <button mat-icon-button (click)="deleteSearch(q)" aria-label="Delete"><mat-icon>delete</mat-icon></button>
          </div></mat-card>
        } @empty { <p class="muted">No schedules yet. Without one, sources still sync on their own interval.</p> }
      </mat-tab>
      <mat-tab label="Execution history">
        <div class="table-wrap"><table mat-table [dataSource]="runs()" class="full">
          <ng-container matColumnDef="kind"><th mat-header-cell *matHeaderCellDef>Task</th><td mat-cell *matCellDef="let r">{{ r.kind.replace('_', ' ') }}</td></ng-container>
          <ng-container matColumnDef="status"><th mat-header-cell *matHeaderCellDef>Status</th>
            <td mat-cell *matCellDef="let r"><span class="badge" [class.ok]="r.status === 'succeeded'" [class.danger]="r.status === 'failed' || r.status === 'interrupted'">{{ r.status }}</span></td></ng-container>
          <ng-container matColumnDef="started"><th mat-header-cell *matHeaderCellDef>Started</th><td mat-cell *matCellDef="let r">{{ r.started_at | date: 'short' }}</td></ng-container>
          <ng-container matColumnDef="duration"><th mat-header-cell *matHeaderCellDef>Duration</th><td mat-cell *matCellDef="let r">{{ r.duration_seconds ?? '…' }}s</td></ng-container>
          <ng-container matColumnDef="detail"><th mat-header-cell *matHeaderCellDef>Details</th>
            <td mat-cell *matCellDef="let r" class="small">{{ r.error || summary(r.stats) }}</td></ng-container>
          <tr mat-header-row *matHeaderRowDef="runCols"></tr><tr mat-row *matRowDef="let r; columns: runCols"></tr>
        </table></div>
      </mat-tab>
    </mat-tab-group>
  }`,
})
export class SettingsComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private dialog = inject(MatDialog);
  s = signal<AutomationSettings | null>(null);
  scoring = signal<ScoringConfig | null>(null);
  searches = signal<ScheduledSearch[]>([]);
  runs = signal<AutomationRun[]>([]);
  sources = signal<JobSource[]>([]);
  runCols = ['kind', 'status', 'started', 'duration', 'detail'];
  modes = [
    { value: 'discovery', icon: 'search', title: 'Discovery only', text: 'Find and score jobs and show the official link. Nothing is ever submitted.' },
    { value: 'assisted', icon: 'handshake', title: 'Assisted application', text: 'Prepare tailored documents and answers; you review and submit on the official page.' },
    { value: 'automated', icon: 'bolt', title: 'Authorized automated submission', text: 'Submit through documented employer APIs only, with your consent, limits and duplicate checks.' },
  ];
  weights = [
    { key: 'w_required_skills', label: 'Required skills' }, { key: 'w_preferred_skills', label: 'Preferred skills' },
    { key: 'w_experience', label: 'Experience' }, { key: 'w_role_alignment', label: 'Role & seniority' },
    { key: 'w_semantic', label: 'Semantic similarity' }, { key: 'w_location', label: 'Location & work model' },
    { key: 'w_salary', label: 'Salary' }, { key: 'w_employment_type', label: 'Employment type' },
  ];

  ngOnInit(): void {
    forkJoin({
      s: this.api.get<AutomationSettings>('automation/settings/'), scoring: this.api.get<ScoringConfig>('scoring-config/'),
      searches: this.api.get<Paginated<ScheduledSearch>>('scheduled-searches/'), runs: this.api.get<Paginated<AutomationRun>>('automation/runs/', { page_size: 50 }),
      sources: this.api.get<Paginated<JobSource>>('sources/', { page_size: 100 }),
    }).subscribe({
      next: r => { this.s.set(r.s); this.scoring.set(r.scoring); this.searches.set(r.searches.results); this.runs.set(r.runs.results); this.sources.set(r.sources.results); },
      error: e => this.notify.error(e),
    });
  }

  summary(stats: Record<string, unknown>): string {
    return Object.entries(stats || {}).filter(([, v]) => typeof v !== 'object').map(([k, v]) => `${k}: ${v}`).join(' · ');
  }

  setMode(mode: string): void {
    const s = this.s()!;
    if (mode === s.application_mode) return;
    if (mode !== 'automated') { this.patch({ application_mode: mode }); return; }
    this.dialog.open(ConfirmDialogComponent, { width: '560px', data: { title: 'Allow authorized automated submission?', confirm: 'I consent',
      message: 'JobPilot may submit applications on your behalf only through documented employer APIs, using resumes and answers you approved, '
        + 'within your daily limit, never twice for the same job, and never answering legal, demographic, disability or sponsorship questions without your review. '
        + 'You can revoke consent at any time.' } })
      .afterClosed().subscribe(ok => ok && this.patch({ application_mode: 'automated', automated_submission_consent: true }));
  }
  revoke(): void { this.patch({ application_mode: 'assisted', automated_submission_consent: false }); }

  save(): void {
    const { automated_submission_consent_at, automated_submission_allowed, ...body } = this.s()! as any;
    this.patch(body);
  }
  private patch(body: Record<string, unknown>): void {
    this.api.patch<AutomationSettings>('automation/settings/', body).subscribe({ next: s => { this.s.set(s); this.notify.success('Settings saved.'); },
      error: e => this.notify.error(e) });
  }

  saveScoring(): void {
    const { updated_at, ...body } = this.scoring()! as any;
    this.api.patch<ScoringConfig>('scoring-config/', body).subscribe({
      next: c => { this.scoring.set(c); this.api.post('matches/rematch/').subscribe(); this.notify.success('Saved. Scores are being recomputed.'); },
      error: e => this.notify.error(e) });
  }

  editSearch(q?: ScheduledSearch): void {
    this.dialog.open(FormDialogComponent, { width: '520px', data: { title: q ? 'Edit schedule' : 'New schedule', value: q ?? { interval_hours: 12, enabled: true }, fields: [
      { key: 'name', label: 'Name', required: true }, { key: 'interval_hours', label: 'Every (hours)', type: 'number', required: true },
      { key: 'enabled', label: 'Enabled', type: 'checkbox' }] } }).afterClosed().subscribe(v => {
      if (!v) return;
      const req = q ? this.api.patch(`scheduled-searches/${q.id}/`, v) : this.api.post('scheduled-searches/', { ...v, sources: [] });
      req.subscribe({ next: () => this.ngOnInit(), error: e => this.notify.error(e) });
    });
  }
  patchSearch(q: ScheduledSearch, body: Partial<ScheduledSearch>): void {
    this.api.patch(`scheduled-searches/${q.id}/`, body).subscribe({ error: e => this.notify.error(e) });
  }
  runNow(q: ScheduledSearch): void {
    this.api.post(`scheduled-searches/${q.id}/run-now/`).subscribe({ next: () => this.notify.success('Search started.'), error: e => this.notify.error(e) });
  }
  deleteSearch(q: ScheduledSearch): void {
    this.api.delete(`scheduled-searches/${q.id}/`).subscribe({ next: () => this.searches.update(l => l.filter(x => x.id !== q.id)), error: e => this.notify.error(e) });
  }
}
