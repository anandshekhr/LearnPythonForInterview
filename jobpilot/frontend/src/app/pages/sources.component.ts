import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatDialog } from '@angular/material/dialog';
import { filter, forkJoin, interval } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Capabilities, Company, JobSource, Paginated } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { FormDialogComponent } from '../shared/form-dialog.component';
import { MATERIAL } from '../shared/material';
import { ConfirmDialogComponent, EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

interface SourceType { source_type: string; label: string; capabilities: Capabilities; }

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Companies & sources" subtitle="Official, publicly documented job-board APIs only.">
    <button mat-stroked-button (click)="addCompany()"><mat-icon>add_business</mat-icon> Add company</button>
    <button mat-flat-button color="primary" (click)="addSource()" [disabled]="!companies().length"><mat-icon>add_link</mat-icon> Add job source</button>
  </jp-page-header>
  <mat-expansion-panel class="info-panel">
    <mat-expansion-panel-header><mat-panel-title><mat-icon>info</mat-icon>&nbsp;Supported sources and how to find the identifier</mat-panel-title></mat-expansion-panel-header>
    @for (t of types(); track t.source_type) {
      <div class="sub-card"><strong>{{ t.label }}</strong> — <a [href]="t.capabilities.documentation_url" target="_blank" rel="noopener">documentation</a>
        <ul>@for (l of t.capabilities.limitations; track l) { <li class="small">{{ l }}</li> }</ul>
        <div class="small muted">{{ hint(t.source_type) }}</div></div>
    }
  </mat-expansion-panel>
  <jp-loading [loading]="loading()" />
  <h2 class="section-title">Job sources</h2>
  @for (s of sources(); track s.id) {
    <mat-card class="item-card">
      <div class="row wrap">
        <span class="dot" [class]="'dot-' + s.status"></span>
        <div class="grow"><div class="strong">{{ s.company_name }} <span class="muted">({{ s.source_type }}: {{ s.identifier }})</span></div>
          <div class="muted small">{{ s.status }} · {{ s.active_job_count }} active jobs · last success {{ s.last_success_at ? (s.last_success_at | date: 'short') : 'never' }}
            · every {{ s.sync_interval_minutes / 60 }} h</div>
          @if (s.last_error) { <div class="error-text small">{{ s.last_error }}</div> }
          <div class="small">
            <span class="cap" [class.on]="s.capabilities.list_jobs">listings</span>
            <span class="cap" [class.on]="s.capabilities.salary_data">salary</span>
            <span class="cap" [class.on]="s.capabilities.application_questions">questions</span>
            <span class="cap" [class.on]="s.capabilities.api_submission_active" [matTooltip]="s.capabilities.api_submission_requires">API submission {{ s.capabilities.api_submission_active ? 'active' : 'needs employer key' }}</span>
          </div>
        </div>
        <mat-slide-toggle [checked]="s.enabled" [disabled]="!s.can_edit" (change)="toggle(s, $event.checked)">Enabled</mat-slide-toggle>
        <button mat-stroked-button (click)="test(s)"><mat-icon>network_check</mat-icon> Test</button>
        <button mat-flat-button color="primary" (click)="sync(s)" [disabled]="!s.enabled"><mat-icon>sync</mat-icon> Sync now</button>
        @if (s.can_edit) { <button mat-icon-button (click)="removeSource(s)" aria-label="Delete source"><mat-icon>delete</mat-icon></button> }
      </div>
    </mat-card>
  } @empty { @if (!loading()) { <jp-empty icon="link_off" title="No job sources" text="Add a company, then connect its Greenhouse or Lever job board." /> } }
  <h2 class="section-title">Company registry</h2>
  <div class="table-wrap"><table mat-table [dataSource]="companies()" class="full">
    <ng-container matColumnDef="name"><th mat-header-cell *matHeaderCellDef>Company</th><td mat-cell *matCellDef="let c"><div class="strong">{{ c.name }}</div>
      @if (c.careers_url) { <a class="small" [href]="c.careers_url" target="_blank" rel="noopener">careers site</a> }</td></ng-container>
    <ng-container matColumnDef="type"><th mat-header-cell *matHeaderCellDef>Type</th><td mat-cell *matCellDef="let c">{{ c.company_type || '—' }}</td></ng-container>
    <ng-container matColumnDef="sources"><th mat-header-cell *matHeaderCellDef>Sources</th><td mat-cell *matCellDef="let c">{{ c.source_count }}</td></ng-container>
    <ng-container matColumnDef="jobs"><th mat-header-cell *matHeaderCellDef>Active jobs</th><td mat-cell *matCellDef="let c">{{ c.active_jobs }}</td></ng-container>
    <ng-container matColumnDef="actions"><th mat-header-cell *matHeaderCellDef></th><td mat-cell *matCellDef="let c">
      @if (c.can_edit) { <button mat-icon-button (click)="addCompany(c)" aria-label="Edit"><mat-icon>edit</mat-icon></button> }</td></ng-container>
    <tr mat-header-row *matHeaderRowDef="cols"></tr><tr mat-row *matRowDef="let c; columns: cols"></tr>
  </table></div>`,
})
export class SourcesComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private dialog = inject(MatDialog);
  loading = signal(true);
  companies = signal<Company[]>([]);
  sources = signal<JobSource[]>([]);
  types = signal<SourceType[]>([]);
  cols = ['name', 'type', 'sources', 'jobs', 'actions'];

  private destroyRef = inject(DestroyRef);
  private pendingUntil = 0;

  ngOnInit(): void {
    this.load();
    // While a synchronisation is pending (new source or "Sync now"), refresh every few seconds.
    interval(3000).pipe(filter(() => Date.now() < this.pendingUntil || this.sources().some(s => s.status === 'unverified' && s.enabled)),
      takeUntilDestroyed(this.destroyRef)).subscribe(() => this.load());
  }

  load(): void {
    forkJoin({
      companies: this.api.get<Paginated<Company>>('companies/', { page_size: 100 }),
      sources: this.api.get<Paginated<JobSource>>('sources/', { page_size: 100 }),
      types: this.api.get<SourceType[]>('sources/types/'),
    }).subscribe({
      next: r => { this.companies.set(r.companies.results); this.sources.set(r.sources.results); this.types.set(r.types); this.loading.set(false); },
      error: e => { this.loading.set(false); this.notify.error(e); },
    });
  }

  hint(type: string): string {
    return type === 'greenhouse' ? 'Identifier = board token: the part after boards.greenhouse.io/ (e.g. "acme" for boards.greenhouse.io/acme).'
      : type === 'lever' ? 'Identifier = site name: the part after jobs.lever.co/ (e.g. "acme"). Use region "eu" for jobs.eu.lever.co.' : '';
  }

  addCompany(c?: Company): void {
    this.dialog.open(FormDialogComponent, { width: '560px', data: { title: c ? 'Edit company' : 'Add company', value: c, fields: [
      { key: 'name', label: 'Name', required: true }, { key: 'website', label: 'Website', type: 'url' },
      { key: 'careers_url', label: 'Careers page', type: 'url' }, { key: 'industry', label: 'Industry' },
      { key: 'company_type', label: 'Type', type: 'select', options: ['Product', 'SaaS', 'Technology', 'AI startup', 'IT services', 'Other'].map(v => ({ value: v, label: v })) },
      { key: 'headquarters', label: 'Headquarters' }, { key: 'notes', label: 'Notes', type: 'textarea' }] } })
      .afterClosed().subscribe(v => v && (c ? this.api.patch(`companies/${c.id}/`, v) : this.api.post('companies/', v)).subscribe({
        next: () => { this.notify.success('Saved.'); this.load(); }, error: e => this.notify.error(e) }));
  }

  addSource(): void {
    this.dialog.open(FormDialogComponent, { width: '560px', data: { title: 'Add job source', value: { source_type: 'greenhouse', sync_interval_minutes: 360 }, fields: [
      { key: 'company', label: 'Company', type: 'select', required: true, options: this.companies().map(c => ({ value: c.id as any, label: c.name })) },
      { key: 'source_type', label: 'Source type', type: 'select', required: true, options: this.types().map(t => ({ value: t.source_type, label: t.label })) },
      { key: 'identifier', label: 'Board token / site name', required: true, hint: 'See “Supported sources” above' },
      { key: 'sync_interval_minutes', label: 'Sync interval (minutes, min 60)', type: 'number' }] } })
      .afterClosed().subscribe(v => v && this.api.post('sources/', v).subscribe({
        next: () => { this.notify.success('Source added. The first synchronisation has started.'); this.pendingUntil = Date.now() + 60000; this.load(); }, error: e => this.notify.error(e) }));
  }

  toggle(s: JobSource, enabled: boolean): void {
    this.api.patch(`sources/${s.id}/`, { enabled }).subscribe({ next: () => this.load(), error: e => this.notify.error(e) });
  }
  sync(s: JobSource): void {
    this.api.post(`sources/${s.id}/sync/`).subscribe({ next: () => { this.notify.success('Synchronisation queued.'); this.pendingUntil = Date.now() + 60000; },
      error: e => this.notify.error(e) });
  }
  test(s: JobSource): void {
    this.api.post<{ ok: boolean; published_jobs?: number; error?: string }>(`sources/${s.id}/test/`).subscribe({
      next: r => r.ok ? this.notify.success(`Reachable: ${r.published_jobs} published jobs.`) : this.notify.error(null, `Not reachable: ${r.error}`),
      error: e => this.notify.error(e) });
  }
  removeSource(s: JobSource): void {
    this.dialog.open(ConfirmDialogComponent, { data: { title: 'Delete source?', message: `All jobs from ${s.company_name} (${s.identifier}) will be removed. Applications keep their history.`, confirm: 'Delete', danger: true } })
      .afterClosed().subscribe(ok => ok && this.api.delete(`sources/${s.id}/`).subscribe({ next: () => this.load(), error: e => this.notify.error(e) }));
  }
}
