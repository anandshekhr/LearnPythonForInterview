import { Component, OnInit, inject, signal } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { PageEvent } from '@angular/material/paginator';
import { Sort } from '@angular/material/sort';
import { ActivatedRoute } from '@angular/router';
import { debounceTime } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Application, Paginated, STATUS_LABELS } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent, ScoreBadgeComponent, StatusChipComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent, StatusChipComponent, ScoreBadgeComponent],
  template: `
  <jp-page-header title="Applications" subtitle="Every application, its status and history.">
    <button mat-stroked-button (click)="api.download('analytics/applications.csv', 'applications.csv')"><mat-icon>download</mat-icon> Export CSV</button>
  </jp-page-header>
  <form class="filters" [formGroup]="filters">
    <mat-form-field appearance="outline" class="grow"><mat-label>Search company, title, notes</mat-label><input matInput formControlName="search"></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Status</mat-label>
      <mat-select formControlName="status"><mat-option value="">Any</mat-option>
        @for (s of statuses; track s.key) { <mat-option [value]="s.key">{{ s.label }}</mat-option> }</mat-select></mat-form-field>
  </form>
  <jp-loading [loading]="loading()" />
  <div class="table-wrap">
    <table mat-table [dataSource]="apps()" matSort (matSortChange)="sort($event)" class="full">
      <ng-container matColumnDef="company"><th mat-header-cell *matHeaderCellDef mat-sort-header="company_name">Company / role</th>
        <td mat-cell *matCellDef="let a"><div class="strong">{{ a.company_name }}</div><div class="muted">{{ a.job_title }}</div></td></ng-container>
      <ng-container matColumnDef="score"><th mat-header-cell *matHeaderCellDef mat-sort-header="match_score">Match</th>
        <td mat-cell *matCellDef="let a"><jp-score [score]="a.match_score" /></td></ng-container>
      <ng-container matColumnDef="status"><th mat-header-cell *matHeaderCellDef>Status</th><td mat-cell *matCellDef="let a"><jp-status [status]="a.status" /></td></ng-container>
      <ng-container matColumnDef="submitted"><th mat-header-cell *matHeaderCellDef mat-sort-header="submitted_at">Submitted</th>
        <td mat-cell *matCellDef="let a">{{ a.submitted_at ? (a.submitted_at | date: 'mediumDate') : '—' }}</td></ng-container>
      <ng-container matColumnDef="next"><th mat-header-cell *matHeaderCellDef>Next</th>
        <td mat-cell *matCellDef="let a">@if (a.next_interview) { <mat-icon inline>event</mat-icon> {{ a.next_interview | date: 'd MMM HH:mm' }} }
          @else if (a.follow_up_date) { follow up {{ a.follow_up_date | date: 'd MMM' }} }</td></ng-container>
      <ng-container matColumnDef="updated"><th mat-header-cell *matHeaderCellDef mat-sort-header="updated_at">Updated</th>
        <td mat-cell *matCellDef="let a">{{ a.updated_at | date: 'mediumDate' }}</td></ng-container>
      <tr mat-header-row *matHeaderRowDef="cols"></tr>
      <tr mat-row *matRowDef="let a; columns: cols" class="clickable" [routerLink]="['/applications', a.id]"></tr>
    </table>
  </div>
  @if (!loading() && !apps().length) { <jp-empty icon="assignment" title="No applications" text="Shortlist a job or tailor a resume to start an application." /> }
  <mat-paginator [length]="count()" [pageSize]="20" [pageIndex]="page() - 1" (page)="pageChange($event)" />`,
})
export class ApplicationsComponent implements OnInit {
  api = inject(ApiService);
  private notify = inject(NotifyService);
  loading = signal(true);
  apps = signal<Application[]>([]);
  count = signal(0);
  page = signal(1);
  ordering = '-updated_at';
  cols = ['company', 'score', 'status', 'submitted', 'next', 'updated'];
  statuses = Object.entries(STATUS_LABELS).filter(([k]) => k !== 'submitting').map(([key, label]) => ({ key, label }));
  filters = inject(FormBuilder).nonNullable.group({ search: [''], status: [inject(ActivatedRoute).snapshot.queryParamMap.get('status') ?? ''] });

  ngOnInit(): void {
    this.filters.valueChanges.pipe(debounceTime(300)).subscribe(() => { this.page.set(1); this.load(); });
    this.load();
  }
  load(): void {
    this.loading.set(true);
    const f = this.filters.getRawValue();
    this.api.get<Paginated<Application>>('applications/', { ...f, ordering: this.ordering, page: this.page() }).subscribe({
      next: r => { this.apps.set(r.results); this.count.set(r.count); this.loading.set(false); },
      error: e => { this.loading.set(false); this.notify.error(e); } });
  }
  sort(s: Sort): void { this.ordering = s.direction ? `${s.direction === 'desc' ? '-' : ''}${s.active}` : '-updated_at'; this.load(); }
  pageChange(e: PageEvent): void { this.page.set(e.pageIndex + 1); this.load(); }
}
