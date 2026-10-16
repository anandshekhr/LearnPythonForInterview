import { DecimalPipe, TitleCasePipe } from '@angular/common';
import { Component, OnInit, inject, input, signal } from '@angular/core';
import { Router } from '@angular/router';
import { filter, interval, map, switchMap, take, timeout } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Job, JobMatch, Paginated, ResumeVersion } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { LoadingComponent, PageHeaderComponent, ScoreBadgeComponent, salaryRange } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, TitleCasePipe, DecimalPipe, PageHeaderComponent, LoadingComponent, ScoreBadgeComponent],
  template: `
  <jp-loading [loading]="!job()" />
  @if (job(); as j) {
    <jp-page-header [title]="j.title" [subtitle]="j.company_name + ' · ' + (j.location_raw || j.work_model)">
      <a mat-stroked-button [href]="j.official_url" target="_blank" rel="noopener noreferrer"><mat-icon>open_in_new</mat-icon> Official posting</a>
      @if (j.application_id) {
        <a mat-flat-button color="primary" [routerLink]="['/applications', j.application_id]"><mat-icon>assignment</mat-icon> View application</a>
      } @else {
        <button mat-stroked-button (click)="track()" [disabled]="j.status !== 'active'"><mat-icon>bookmark_add</mat-icon> Shortlist</button>
      }
      <button mat-flat-button color="primary" (click)="generate()" [disabled]="busy() || j.status !== 'active'">
        <mat-icon>auto_fix_high</mat-icon> {{ busy() ? 'Generating…' : 'Tailor resume' }}</button>
    </jp-page-header>
    @if (j.status === 'expired') { <mat-card class="callout warn"><mat-icon>event_busy</mat-icon> This posting expired on {{ j.expired_at | date }}.</mat-card> }
    @if (j.duplicate_of) { <mat-card class="callout"><mat-icon>content_copy</mat-icon> This role is also listed elsewhere. <a [routerLink]="['/jobs', j.duplicate_of]">View the original</a></mat-card> }
    <div class="grid-2 wide-left">
      <div>
        <mat-card class="item-card">
          <div class="facts">
            <div><span class="muted">Work model</span><div>{{ j.work_model }}</div></div>
            <div><span class="muted">Employment</span><div>{{ j.employment_type || '—' }}</div></div>
            <div><span class="muted">Seniority</span><div>{{ j.seniority }}</div></div>
            <div><span class="muted">Salary</span><div>{{ salary(j) || 'Not disclosed' }}</div></div>
            <div><span class="muted">Published</span><div>{{ (j.published_at || j.first_seen_at) | date }}</div></div>
            <div><span class="muted">Source</span><div>{{ j.source_type }}</div></div>
          </div>
        </mat-card>
        @if (j.requirements; as r) {
          <mat-card class="item-card"><h3>Extracted requirements</h3>
            @if (r.min_years) { <p>Experience: {{ r.min_years }}{{ r.max_years ? '–' + r.max_years : '+' }} years</p> }
            <div class="chips">@for (s of r.required_skills; track s) { <span class="chip" [class.ok]="has(s)" [class.missing]="!has(s)">{{ s }}</span> }</div>
            @if (r.preferred_skills.length) { <div class="muted small">Nice to have:</div>
              <div class="chips">@for (s of r.preferred_skills; track s) { <span class="chip soft" [class.ok]="has(s)">{{ s }}</span> }</div> }
          </mat-card>
        }
        <mat-card class="item-card"><h3>Description</h3>
          <div class="description">{{ j.description_text }}</div>
          <p class="muted small">Job descriptions come from the employer's site and are shown as plain text.</p>
        </mat-card>
      </div>
      <div>
        @if (match(); as m) {
          <mat-card class="item-card">
            <div class="row"><jp-score [score]="m.score" [category]="m.category" /><h3 class="grow">{{ m.category | titlecase }} match</h3></div>
            <p>{{ m.explanation }}</p>
            <p class="strong">{{ m.recommended_action }}</p>
            @if (m.excluded_reason) { <p class="error-text">{{ m.excluded_reason }}</p> }
            <h4>Score breakdown</h4>
            @for (c of components(m); track c.name) {
              <div class="signal">
                <div class="row"><span class="grow">{{ c.name }}</span><span class="muted">weight {{ c.weight }}</span>
                  <span class="strong">{{ c.score === null ? 'n/a' : (c.score * 100 | number: '1.0-0') + '%' }}</span></div>
                <mat-progress-bar [value]="(c.score ?? 0) * 100" [class.unknown]="c.score === null" />
                <div class="muted small">{{ c.detail }}</div>
              </div>
            }
            <p class="muted small">Information coverage {{ m.information_coverage * 100 | number: '1.0-0' }}%.</p>
            @if (m.gaps.length) { <h4>Gaps</h4><ul>@for (g of m.gaps; track g) { <li>{{ g }}</li> }</ul> }
            @if (m.skills_unverified.length) { <p class="muted small">Listed but unverified in your profile: {{ m.skills_unverified.join(', ') }}. <a routerLink="/profile">Verify them</a> only if accurate.</p> }
            @if (m.relevant_experience.length) {
              <h4>Relevant experience</h4>
              <ul>@for (r of m.relevant_experience; track r.label) { <li><strong>{{ r.label }}</strong> — {{ r.skills.join(', ') }}</li> }</ul>
            }
            <p class="muted small">{{ m.location_fit }} · {{ m.salary_fit }}</p>
          </mat-card>
        } @else { <mat-card class="item-card"><p class="muted">No match score yet. Verify your profile to get one.</p></mat-card> }
        @if (versions().length) {
          <mat-card class="item-card"><h3>Tailored resumes for this job</h3>
            @for (v of versions(); track v.id) {
              <a class="row-link" [routerLink]="['/resume-versions', v.id]"><mat-icon>description</mat-icon>
                <span class="grow">v{{ v.version_number }} · {{ v.created_at | date: 'short' }}</span><span class="badge">{{ v.status.replace('_', ' ') }}</span></a>
            }
          </mat-card>
        }
      </div>
    </div>
  }`,
})
export class JobDetailComponent implements OnInit {
  id = input.required<string>();
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private router = inject(Router);
  job = signal<Job | null>(null);
  match = signal<JobMatch | null>(null);
  versions = signal<ResumeVersion[]>([]);
  busy = signal(false);

  ngOnInit(): void {
    this.api.get<Job>(`jobs/${this.id()}/`).subscribe({ next: j => this.job.set(j), error: e => this.notify.error(e, 'Job not found.') });
    this.api.get<Paginated<JobMatch>>('matches/', { job: this.id() }).subscribe(r => this.match.set(r.results[0] ?? null));
    this.api.get<Paginated<ResumeVersion>>('resume-versions/', { job: this.id() }).subscribe(r => this.versions.set(r.results));
  }

  has(skill: string): boolean { return this.match()?.skills_matched.includes(skill) ?? false; }
  salary(j: Job): string { return salaryRange(j.salary_min, j.salary_max, j.salary_currency) || j.salary_text; }
  components(m: JobMatch) {
    return Object.entries(m.components).map(([name, c]) => ({ name: name.replace(/_/g, ' '), ...c })).sort((a, b) => b.weight - a.weight);
  }

  track(): void {
    this.api.post<{ id: number }>('applications/', { job: this.job()!.id, status: 'shortlisted' }).subscribe({
      next: a => { this.notify.success('Added to your applications.'); this.job.update(j => j && { ...j, application_id: a.id }); },
      error: e => this.notify.error(e),
    });
  }

  generate(): void {
    this.busy.set(true);
    this.api.post<{ result?: { status: string; resume_version_id?: number; error?: string } }>('documents/generate/', { job: this.job()!.id, include_cover_letter: true })
      .subscribe({
        next: r => {
          this.busy.set(false);
          if (r.result?.status === 'ok') this.router.navigate(['/resume-versions', r.result.resume_version_id]);
          else if (r.result?.status === 'failed') this.notify.error(null, r.result.error);
          else this.waitForVersion();
        },
        error: e => { this.busy.set(false); this.notify.error(e); },
      });
  }

  /** Generation runs on a Celery worker: poll until the new version exists, then open it. */
  private waitForVersion(): void {
    this.busy.set(true);
    const known = new Set(this.versions().map(v => v.id));
    interval(2000).pipe(
      switchMap(() => this.api.get<Paginated<ResumeVersion>>('resume-versions/', { job: this.id() })),
      map(r => r.results.find(v => !known.has(v.id))), filter(v => !!v), take(1), timeout(90000),
    ).subscribe({
      next: v => { this.busy.set(false); this.router.navigate(['/resume-versions', v!.id]); },
      error: () => { this.busy.set(false); this.notify.success('Still generating. It will appear under Tailored resumes.'); },
    });
  }
}
