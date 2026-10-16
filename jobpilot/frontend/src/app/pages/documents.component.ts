import { Component, OnInit, inject, signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ApiService } from '../core/api.service';
import { Paginated, ResumeVersion } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Tailored resumes" subtitle="Every tailored resume needs your approval before it can be used." />
  <mat-button-toggle-group [value]="status()" (change)="status.set($event.value); load()" aria-label="Status">
    <mat-button-toggle value="awaiting_approval">Awaiting approval</mat-button-toggle>
    <mat-button-toggle value="approved">Approved</mat-button-toggle><mat-button-toggle value="rejected">Rejected</mat-button-toggle>
    <mat-button-toggle value="">All</mat-button-toggle>
  </mat-button-toggle-group>
  <jp-loading [loading]="loading()" />
  @for (v of versions(); track v.id) {
    <a class="row-link card-like" [routerLink]="['/resume-versions', v.id]">
      <mat-icon>description</mat-icon>
      <div class="grow"><div class="strong">{{ v.job_title || v.label }}</div><div class="muted">{{ v.company_name }} · v{{ v.version_number }} · {{ v.created_at | date: 'medium' }}</div></div>
      @if (v.tailoring_notes.warnings?.length) { <mat-icon matTooltip="Has warnings to read" class="warn-icon">warning</mat-icon> }
      <span class="badge">{{ v.status.replace('_', ' ') }}</span>
    </a>
  } @empty { @if (!loading()) { <jp-empty icon="description" title="No tailored resumes" text="Open a job and choose “Tailor resume”." /> } }`,
})
export class DocumentsComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  loading = signal(true);
  versions = signal<ResumeVersion[]>([]);
  status = signal(inject(ActivatedRoute).snapshot.queryParamMap.get('status') ?? 'awaiting_approval');

  ngOnInit(): void { this.load(); }
  load(): void {
    this.loading.set(true);
    this.api.get<Paginated<ResumeVersion>>('resume-versions/', { kind: 'tailored', status: this.status(), page_size: 50 }).subscribe({
      next: r => { this.versions.set(r.results); this.loading.set(false); }, error: e => { this.loading.set(false); this.notify.error(e); } });
  }
}
