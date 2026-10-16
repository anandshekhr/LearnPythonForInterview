import { Component, OnInit, inject, input, signal } from '@angular/core';
import { Router } from '@angular/router';
import { ApiService } from '../core/api.service';
import { ParsedResume, Resume } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { LoadingComponent, PageHeaderComponent } from '../shared/ui';

/** Review the parsed draft before it becomes verified profile data. Everything is editable. */
@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent],
  template: `
  <jp-page-header title="Review extracted details" subtitle="Correct anything that is wrong. Only what you confirm here is used.">
    <a mat-button routerLink="/resumes">Cancel</a>
    <button mat-flat-button color="primary" (click)="apply()" [disabled]="!draft() || saving()"><mat-icon>verified</mat-icon> Confirm & apply to profile</button>
  </jp-page-header>
  <jp-loading [loading]="!draft()" />
  @if (draft(); as d) {
    @if (d.warnings.length) { <mat-card class="callout warn"><mat-icon>warning</mat-icon><div>@for (w of d.warnings; track w) {<div>{{ w }}</div>}</div></mat-card> }
    <p class="muted">Parser: {{ d.parser }}. Applying replaces your experience, education, projects and certifications with this list and merges skills.</p>
    <mat-card class="item-card"><h3>Contact & summary</h3>
      <div class="form-grid">
        @for (k of contactKeys; track k) {
          <mat-form-field appearance="outline"><mat-label>{{ k.replace('_', ' ') }}</mat-label><input matInput [(ngModel)]="d.contact[k]"></mat-form-field>
        }
        <mat-form-field appearance="outline"><mat-label>Headline</mat-label><input matInput [(ngModel)]="d.headline"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Total experience (years)</mat-label><input matInput type="number" step="0.5" [(ngModel)]="d.total_experience_years">
          <mat-hint>Computed from dates{{ d.stated_experience_years ? '; resume states ' + d.stated_experience_years + '+' : '' }}</mat-hint></mat-form-field>
        <mat-form-field appearance="outline" class="span-2"><mat-label>Summary</mat-label><textarea matInput rows="3" [(ngModel)]="d.summary"></textarea></mat-form-field>
      </div>
    </mat-card>
    <mat-card class="item-card"><h3>Skills ({{ d.skills.length }})</h3>
      <mat-chip-set>@for (s of d.skills; track s.name; let i = $index) {
        <mat-chip (removed)="d.skills.splice(i, 1)">{{ s.name }}<button matChipRemove [attr.aria-label]="'Remove ' + s.name"><mat-icon>cancel</mat-icon></button></mat-chip>
      }</mat-chip-set>
      <p class="muted">Remove anything you would not want to claim. You can add skills later on the profile page.</p>
    </mat-card>
    <mat-card class="item-card"><div class="row"><h3 class="grow">Experience ({{ d.employment.length }})</h3>
      <button mat-stroked-button (click)="d.employment.push({ title: '', company: '', achievements: [], is_current: false })"><mat-icon>add</mat-icon> Add role</button></div>
      @for (e of d.employment; track $index; let i = $index) {
        <div class="sub-card">
          <div class="form-grid">
            <mat-form-field appearance="outline"><mat-label>Title</mat-label><input matInput [(ngModel)]="e.title"></mat-form-field>
            <mat-form-field appearance="outline"><mat-label>Company</mat-label><input matInput [(ngModel)]="e.company"></mat-form-field>
            <mat-form-field appearance="outline"><mat-label>Start (YYYY-MM-DD)</mat-label><input matInput [(ngModel)]="e.start_date"></mat-form-field>
            <mat-form-field appearance="outline"><mat-label>End (YYYY-MM-DD)</mat-label><input matInput [(ngModel)]="e.end_date" [disabled]="!!e.is_current"></mat-form-field>
            <mat-checkbox [(ngModel)]="e.is_current">Current role</mat-checkbox>
            <button mat-button color="warn" (click)="d.employment.splice(i, 1)"><mat-icon>delete</mat-icon> Remove role</button>
            <mat-form-field appearance="outline" class="span-2"><mat-label>Achievements (one per line)</mat-label>
              <textarea matInput rows="4" [ngModel]="(e.achievements || []).join('\n')" (ngModelChange)="e.achievements = splitLines($event)"></textarea></mat-form-field>
          </div>
        </div>
      }
    </mat-card>
    <mat-card class="item-card"><h3>Education</h3>
      @for (ed of d.education; track $index; let i = $index) {
        <div class="form-grid">
          <mat-form-field appearance="outline"><mat-label>Institution</mat-label><input matInput [(ngModel)]="ed.institution"></mat-form-field>
          <mat-form-field appearance="outline"><mat-label>Degree</mat-label><input matInput [(ngModel)]="ed.degree"></mat-form-field>
          <mat-form-field appearance="outline"><mat-label>End year</mat-label><input matInput type="number" [(ngModel)]="ed.end_year"></mat-form-field>
          <button mat-button color="warn" (click)="d.education.splice(i, 1)">Remove</button>
        </div>
      }
    </mat-card>
    <mat-card class="item-card"><h3>Projects & certifications</h3>
      @for (p of d.projects; track $index; let i = $index) {
        <div class="row"><mat-form-field appearance="outline" class="grow"><mat-label>Project</mat-label><input matInput [(ngModel)]="p.name"></mat-form-field>
          <button mat-icon-button (click)="d.projects.splice(i, 1)" aria-label="Remove project"><mat-icon>delete</mat-icon></button></div>
      }
      @for (c of d.certifications; track $index; let i = $index) {
        <div class="row"><mat-form-field appearance="outline" class="grow"><mat-label>Certification</mat-label><input matInput [(ngModel)]="c.name"></mat-form-field>
          <button mat-icon-button (click)="d.certifications.splice(i, 1)" aria-label="Remove certification"><mat-icon>delete</mat-icon></button></div>
      }
    </mat-card>
  }`,
})
export class ResumeReviewComponent implements OnInit {
  id = input.required<string>();
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private router = inject(Router);
  draft = signal<ParsedResume | null>(null);
  saving = signal(false);
  contactKeys = ['full_name', 'email', 'phone', 'location', 'linkedin_url', 'github_url', 'portfolio_url'];

  ngOnInit(): void {
    this.api.get<Resume>(`resumes/${this.id()}/`).subscribe({
      next: r => this.draft.set(structuredClone(r.parsed_data!)), error: e => this.notify.error(e, 'Could not load the parsed resume.') });
  }

  splitLines(v: string): string[] { return v.split('\n').map(s => s.trim()).filter(Boolean); }

  apply(): void {
    const d = this.draft();
    if (!d) return;
    const bad = d.employment.find(e => !e.title || !e.company);
    if (bad) { this.notify.error(null, 'Every role needs a title and a company.'); return; }
    this.saving.set(true);
    this.api.post(`resumes/${this.id()}/apply/`, { data: d, replace_sections: true }).subscribe({
      next: () => { this.notify.success('Profile updated and verified. Matching will use these details.'); this.api.post('matches/rematch/').subscribe(); this.router.navigate(['/profile']); },
      error: e => { this.saving.set(false); this.notify.error(e); },
    });
  }
}
