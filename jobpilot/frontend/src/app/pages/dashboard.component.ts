import { Component, OnInit, inject, signal } from '@angular/core';
import { forkJoin } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Application, Interview, JobMatch, Overview, Paginated, Profile } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent, ScoreBadgeComponent, StatusChipComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, ScoreBadgeComponent, StatusChipComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Overview" subtitle="Your job search at a glance">
    <a mat-stroked-button routerLink="/recommended"><mat-icon>recommend</mat-icon> Recommended jobs</a>
  </jp-page-header>
  <jp-loading [loading]="loading()" />
  @if (profile(); as p) {
    @if (!p.verified_at) {
      <mat-card class="callout warn"><mat-icon>info</mat-icon>
        <div><strong>Set up your profile first.</strong> Upload your resume and confirm the extracted details so jobs can be matched.
        </div><a mat-flat-button color="primary" routerLink="/resumes">Upload resume</a></mat-card>
    }
  }
  @if (o(); as o) {
    <section class="metric-grid">
      @for (m of metrics(o); track m.label) {
        <a class="metric" [routerLink]="m.link" [queryParams]="m.params">
          <mat-icon>{{ m.icon }}</mat-icon><div class="value">{{ m.value }}</div><div class="label">{{ m.label }}</div>
        </a>
      }
    </section>
    <section class="metric-grid small">
      @for (c of conversion(o); track c.label) {
        <div class="metric"><div class="value">{{ c.value === null ? '—' : c.value + '%' }}</div><div class="label">{{ c.label }}</div></div>
      }
    </section>
  }
  <div class="grid-2">
    <mat-card>
      <mat-card-header><mat-card-title>Top matches</mat-card-title></mat-card-header>
      <mat-card-content>
        @for (m of matches(); track m.id) {
          <a class="row-link" [routerLink]="['/jobs', m.job.id]">
            <jp-score [score]="m.score" [category]="m.category" />
            <div class="grow"><div class="strong">{{ m.job.title }}</div><div class="muted">{{ m.job.company_name }} · {{ m.job.location_raw || m.job.work_model }}</div></div>
          </a>
        } @empty { <jp-empty icon="search" title="No matches yet" text="Add companies and sources to start discovering jobs."><a mat-button routerLink="/sources">Add sources</a></jp-empty> }
      </mat-card-content>
    </mat-card>
    <mat-card>
      <mat-card-header><mat-card-title>Needs your attention</mat-card-title></mat-card-header>
      <mat-card-content>
        @for (a of attention(); track a.id) {
          <a class="row-link" [routerLink]="['/applications', a.id]">
            <jp-status [status]="a.status" />
            <div class="grow"><div class="strong">{{ a.job_title }}</div><div class="muted">{{ a.company_name }}</div></div>
          </a>
        } @empty { <jp-empty icon="task_alt" title="All caught up" text="No applications are waiting on you." /> }
      </mat-card-content>
    </mat-card>
    <mat-card>
      <mat-card-header><mat-card-title>Upcoming interviews</mat-card-title></mat-card-header>
      <mat-card-content>
        @for (i of interviews(); track i.id) {
          <a class="row-link" [routerLink]="['/applications', i.application]">
            <mat-icon>event</mat-icon>
            <div class="grow"><div class="strong">{{ i.company_name }} — {{ i.round_name }}</div><div class="muted">{{ i.scheduled_at | date: 'EEE d MMM, HH:mm' }} · {{ i.mode }}</div></div>
          </a>
        } @empty { <jp-empty icon="event_available" title="No interviews scheduled" /> }
      </mat-card-content>
    </mat-card>
  </div>`,
})
export class DashboardComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  loading = signal(true);
  o = signal<Overview | null>(null);
  profile = signal<Profile | null>(null);
  matches = signal<JobMatch[]>([]);
  attention = signal<Application[]>([]);
  interviews = signal<Interview[]>([]);

  ngOnInit(): void {
    forkJoin({
      overview: this.api.get<Overview>('analytics/overview/'),
      profile: this.api.get<Profile>('profile/'),
      matches: this.api.get<Paginated<JobMatch>>('matches/', { page_size: 6, job_status: 'active', hide_excluded: true, ordering: '-score' }),
      apps: this.api.get<Paginated<Application>>('applications/', { status__in: 'awaiting_approval,ready_to_apply,submission_failed', page_size: 6 }),
      interviews: this.api.get<Paginated<Interview>>('interviews/', { start: new Date().toISOString(), outcome: 'pending', page_size: 5 }),
    }).subscribe({
      next: r => {
        this.o.set(r.overview); this.profile.set(r.profile); this.matches.set(r.matches.results);
        this.attention.set(r.apps.results); this.interviews.set(r.interviews.results); this.loading.set(false);
      },
      error: e => { this.loading.set(false); this.notify.error(e, 'Could not load the dashboard.'); },
    });
  }

  metrics(o: Overview) {
    return [
      { label: 'Jobs discovered', value: o.jobs_discovered, icon: 'travel_explore', link: '/jobs', params: {} },
      { label: 'New this week', value: o.new_jobs_this_week, icon: 'fiber_new', link: '/jobs', params: { recent: 1 } },
      { label: 'High-match jobs', value: o.high_match_jobs, icon: 'star', link: '/recommended', params: {} },
      { label: 'Awaiting review', value: o.awaiting_review, icon: 'rate_review', link: '/documents', params: { status: 'awaiting_approval' } },
      { label: 'Applications submitted', value: o.applications_submitted, icon: 'send', link: '/applications', params: {} },
      { label: 'Interviews scheduled', value: o.interviews_scheduled, icon: 'event', link: '/interviews', params: {} },
      { label: 'Offers received', value: o.offers_received, icon: 'celebration', link: '/applications', params: { status: 'offer_received' } },
      { label: 'Submission failures', value: o.submission_failures, icon: 'error', link: '/applications', params: { status: 'submission_failed' } },
    ];
  }

  conversion(o: Overview) {
    return [
      { label: 'Response rate', value: o.conversion.response_rate },
      { label: 'Interview rate', value: o.conversion.interview_rate },
      { label: 'Offer rate', value: o.conversion.offer_rate },
      { label: 'Interview → offer', value: o.conversion.interview_to_offer },
    ];
  }
}
