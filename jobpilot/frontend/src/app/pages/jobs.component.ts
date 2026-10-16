import { Component, OnInit, inject, signal } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { PageEvent } from '@angular/material/paginator';
import { Sort } from '@angular/material/sort';
import { ActivatedRoute } from '@angular/router';
import { debounceTime } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Company, Job, Paginated } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent, ScoreBadgeComponent, salaryRange } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, ScoreBadgeComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Job discovery" subtitle="Every job synchronised from your enabled sources." />
  <form class="filters" [formGroup]="filters">
    <mat-form-field appearance="outline" class="grow"><mat-label>Search title, company, description</mat-label>
      <input matInput formControlName="q"><mat-icon matSuffix>search</mat-icon></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Location</mat-label><input matInput formControlName="location"></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Work model</mat-label>
      <mat-select formControlName="work_model"><mat-option value="">Any</mat-option><mat-option value="remote">Remote</mat-option>
        <mat-option value="hybrid">Hybrid</mat-option><mat-option value="onsite">On-site</mat-option><mat-option value="unknown">Unknown</mat-option></mat-select></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Seniority</mat-label>
      <mat-select formControlName="seniority"><mat-option value="">Any</mat-option>
        @for (s of seniorities; track s) { <mat-option [value]="s">{{ s }}</mat-option> }</mat-select></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Company</mat-label>
      <mat-select formControlName="company"><mat-option value="">Any</mat-option>
        @for (c of companies(); track c.id) { <mat-option [value]="c.id">{{ c.name }}</mat-option> }</mat-select></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Min score</mat-label><input matInput type="number" min="0" max="100" formControlName="min_score"></mat-form-field>
    <mat-form-field appearance="outline"><mat-label>Status</mat-label>
      <mat-select formControlName="status"><mat-option value="active">Active</mat-option><mat-option value="expired">Expired</mat-option><mat-option value="">All</mat-option></mat-select></mat-form-field>
    <mat-checkbox formControlName="recent">New in last 7 days</mat-checkbox>
  </form>
  <jp-loading [loading]="loading()" />
  <div class="table-wrap">
    <table mat-table [dataSource]="jobs()" matSort (matSortChange)="sort($event)" class="full">
      <ng-container matColumnDef="score"><th mat-header-cell *matHeaderCellDef mat-sort-header="matches__score">Match</th>
        <td mat-cell *matCellDef="let j"><jp-score [score]="j.my_match?.score" [category]="j.my_match?.category" /></td></ng-container>
      <ng-container matColumnDef="title"><th mat-header-cell *matHeaderCellDef mat-sort-header="title">Role</th>
        <td mat-cell *matCellDef="let j"><div class="strong">{{ j.title }}</div><div class="muted">{{ j.company_name }}</div></td></ng-container>
      <ng-container matColumnDef="location"><th mat-header-cell *matHeaderCellDef>Location</th>
        <td mat-cell *matCellDef="let j">{{ j.location_raw || '—' }}<div class="muted">{{ j.work_model }}</div></td></ng-container>
      <ng-container matColumnDef="salary"><th mat-header-cell *matHeaderCellDef>Salary</th>
        <td mat-cell *matCellDef="let j">{{ salary(j) || '—' }}</td></ng-container>
      <ng-container matColumnDef="published"><th mat-header-cell *matHeaderCellDef mat-sort-header="first_seen_at">Found</th>
        <td mat-cell *matCellDef="let j">{{ (j.published_at || j.first_seen_at) | date: 'mediumDate' }}
          @if (j.status === 'expired') { <span class="badge danger">expired</span> }</td></ng-container>
      <ng-container matColumnDef="state"><th mat-header-cell *matHeaderCellDef></th>
        <td mat-cell *matCellDef="let j">@if (j.application_id) { <span class="badge ok">tracking</span> }</td></ng-container>
      <tr mat-header-row *matHeaderRowDef="cols"></tr>
      <tr mat-row *matRowDef="let j; columns: cols" class="clickable" [routerLink]="['/jobs', j.id]"></tr>
    </table>
  </div>
  @if (!loading() && !jobs().length) {
    <jp-empty icon="travel_explore" title="No jobs found" text="Try different filters, or add companies and sources to discover more jobs.">
      <a mat-button routerLink="/sources">Manage sources</a></jp-empty>
  }
  <mat-paginator [length]="count()" [pageSize]="pageSize" [pageIndex]="page() - 1" [pageSizeOptions]="[20, 50, 100]" (page)="pageChange($event)" />`,
})
export class JobsComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private route = inject(ActivatedRoute);
  loading = signal(true);
  jobs = signal<Job[]>([]);
  companies = signal<Company[]>([]);
  count = signal(0);
  page = signal(1);
  pageSize = 20;
  ordering = '-first_seen_at';
  cols = ['score', 'title', 'location', 'salary', 'published', 'state'];
  seniorities = ['intern', 'junior', 'mid', 'senior', 'lead', 'principal', 'manager'];
  filters = inject(FormBuilder).nonNullable.group({
    q: [''], location: [''], work_model: [''], seniority: [''], company: [''], min_score: [''], status: ['active'], recent: [false],
  });

  ngOnInit(): void {
    if (this.route.snapshot.queryParamMap.get('recent')) this.filters.patchValue({ recent: true });
    this.api.get<Paginated<Company>>('companies/', { page_size: 100 }).subscribe(r => this.companies.set(r.results));
    this.filters.valueChanges.pipe(debounceTime(300)).subscribe(() => { this.page.set(1); this.load(); });
    this.load();
  }

  load(): void {
    this.loading.set(true);
    const f = this.filters.getRawValue();
    const seenAfter = f.recent ? new Date(Date.now() - 7 * 864e5).toISOString() : '';
    this.api.get<Paginated<Job>>('jobs/', {
      q: f.q, location: f.location, work_model: f.work_model, seniority: f.seniority, company: f.company, min_score: f.min_score,
      status: f.status, seen_after: seenAfter, ordering: this.ordering, page: this.page(), page_size: this.pageSize,
    }).subscribe({
      next: r => { this.jobs.set(r.results); this.count.set(r.count); this.loading.set(false); },
      error: e => { this.loading.set(false); this.notify.error(e); },
    });
  }

  sort(s: Sort): void { this.ordering = s.direction ? `${s.direction === 'desc' ? '-' : ''}${s.active}` : '-first_seen_at'; this.load(); }
  pageChange(e: PageEvent): void { this.page.set(e.pageIndex + 1); this.pageSize = e.pageSize; this.load(); }
  salary(j: Job): string { return salaryRange(j.salary_min, j.salary_max, j.salary_currency) || j.salary_text; }
}
