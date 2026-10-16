import { Component, OnInit, inject, signal } from '@angular/core';
import { FormBuilder, Validators } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { forkJoin } from 'rxjs';
import { ApiService } from '../core/api.service';
import { CandidateSkill, Certification, Education, Employment, Paginated, Profile, Project } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { FieldDef, FormDialogComponent } from '../shared/form-dialog.component';
import { MATERIAL } from '../shared/material';
import { ConfirmDialogComponent, EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

type Section = 'employment' | 'education' | 'projects' | 'certifications';

const FIELDS: Record<Section, FieldDef[]> = {
  employment: [
    { key: 'title', label: 'Job title', required: true }, { key: 'company', label: 'Company', required: true },
    { key: 'location', label: 'Location' }, { key: 'start_date', label: 'Start date', type: 'date' },
    { key: 'end_date', label: 'End date', type: 'date' }, { key: 'is_current', label: 'I currently work here', type: 'checkbox' },
    { key: 'description', label: 'Description', type: 'textarea' }, { key: 'achievements', label: 'Achievements', type: 'list' },
    { key: 'technologies', label: 'Technologies', type: 'list' }, { key: 'verified', label: 'Verified (use as evidence)', type: 'checkbox' },
  ],
  education: [
    { key: 'institution', label: 'Institution', required: true }, { key: 'degree', label: 'Degree' },
    { key: 'field_of_study', label: 'Field of study' }, { key: 'start_year', label: 'Start year', type: 'number' },
    { key: 'end_year', label: 'End year', type: 'number' }, { key: 'grade', label: 'Grade' },
    { key: 'verified', label: 'Verified', type: 'checkbox' },
  ],
  projects: [
    { key: 'name', label: 'Name', required: true }, { key: 'role', label: 'Your role' }, { key: 'description', label: 'Description', type: 'textarea' },
    { key: 'highlights', label: 'Highlights', type: 'list' }, { key: 'technologies', label: 'Technologies', type: 'list' },
    { key: 'url', label: 'URL', type: 'url' }, { key: 'verified', label: 'Verified', type: 'checkbox' },
  ],
  certifications: [
    { key: 'name', label: 'Name', required: true }, { key: 'issuer', label: 'Issuer' }, { key: 'issued_on', label: 'Issued on', type: 'date' },
    { key: 'expires_on', label: 'Expires on', type: 'date' }, { key: 'credential_url', label: 'Credential URL', type: 'url' },
    { key: 'verified', label: 'Verified', type: 'checkbox' },
  ],
};

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Career profile" subtitle="Only verified facts are used for matching and tailored resumes.">
    <a mat-stroked-button routerLink="/resumes"><mat-icon>upload_file</mat-icon> Import from resume</a>
  </jp-page-header>
  <jp-loading [loading]="loading()" />
  @if (profile(); as p) {
    <mat-card class="callout" [class.warn]="p.completeness.score < 100">
      <mat-icon>{{ p.completeness.score === 100 ? 'verified' : 'pending' }}</mat-icon>
      <div class="grow"><strong>Profile {{ p.completeness.score }}% complete.</strong>
        @for (c of checkList(p); track c.key) { <span class="check" [class.ok]="c.ok">{{ c.ok ? '✓' : '○' }} {{ c.key }}</span> }
        @if (p.verified_at) { <div class="muted">Last verified {{ p.verified_at | date: 'mediumDate' }}</div> }
      </div>
    </mat-card>
  }
  <mat-tab-group animationDuration="0ms">
    <mat-tab label="Details">
      <form class="form-grid" [formGroup]="details" (ngSubmit)="saveDetails()">
        <mat-form-field appearance="outline"><mat-label>Full name</mat-label><input matInput formControlName="full_name"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Headline</mat-label><input matInput formControlName="headline"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Current title</mat-label><input matInput formControlName="current_title"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Total experience (years)</mat-label>
          <input matInput type="number" step="0.5" min="0" max="60" formControlName="total_experience_years"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Email</mat-label><input matInput type="email" formControlName="email"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Phone</mat-label><input matInput formControlName="phone"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Location</mat-label><input matInput formControlName="location"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>LinkedIn URL</mat-label><input matInput type="url" formControlName="linkedin_url"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>GitHub URL</mat-label><input matInput type="url" formControlName="github_url"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Portfolio URL</mat-label><input matInput type="url" formControlName="portfolio_url"></mat-form-field>
        <mat-form-field appearance="outline" class="span-2"><mat-label>Summary</mat-label><textarea matInput rows="4" formControlName="summary"></textarea></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Delete closed data after (days, 0 = keep)</mat-label>
          <input matInput type="number" min="0" formControlName="data_retention_days"></mat-form-field>
        <div class="span-2 form-actions"><button mat-flat-button color="primary" [disabled]="details.invalid || details.pristine">Save details</button></div>
      </form>
    </mat-tab>

    <mat-tab label="Preferences">
      <form class="form-grid" [formGroup]="prefs" (ngSubmit)="savePrefs()">
        <mat-form-field appearance="outline" class="span-2"><mat-label>Target roles</mat-label>
          <textarea matInput rows="4" formControlName="target_roles"></textarea><mat-hint>One per line</mat-hint></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Preferred locations</mat-label>
          <textarea matInput rows="3" formControlName="preferred_locations"></textarea><mat-hint>One per line, e.g. Noida, Delhi NCR, India</mat-hint></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Company types</mat-label>
          <textarea matInput rows="3" formControlName="company_types"></textarea><mat-hint>One per line</mat-hint></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Work models</mat-label>
          <mat-select multiple formControlName="work_models">
            <mat-option value="remote">Remote</mat-option><mat-option value="hybrid">Hybrid</mat-option><mat-option value="onsite">On-site</mat-option>
          </mat-select></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Employment types</mat-label>
          <mat-select multiple formControlName="employment_types">
            <mat-option value="full_time">Full-time</mat-option><mat-option value="contract">Contract</mat-option>
            <mat-option value="part_time">Part-time</mat-option><mat-option value="internship">Internship</mat-option>
          </mat-select></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Currency</mat-label><input matInput maxlength="3" formControlName="salary_currency"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Minimum annual salary</mat-label>
          <input matInput type="number" min="0" formControlName="salary_min"><mat-hint>{{ lpa(prefs.value.salary_min) }}</mat-hint></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Target maximum annual salary</mat-label>
          <input matInput type="number" min="0" formControlName="salary_max"><mat-hint>{{ lpa(prefs.value.salary_max) }}</mat-hint></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Minimum match score to consider</mat-label>
          <input matInput type="number" min="0" max="100" formControlName="min_match_score"></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Excluded companies</mat-label>
          <textarea matInput rows="3" formControlName="excluded_companies"></textarea><mat-hint>One per line</mat-hint></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Excluded keywords</mat-label>
          <textarea matInput rows="3" formControlName="excluded_keywords"></textarea><mat-hint>One per line</mat-hint></mat-form-field>
        <mat-slide-toggle formControlName="open_to_relocation">Open to relocation</mat-slide-toggle>
        <div class="span-2 form-actions"><button mat-flat-button color="primary" [disabled]="prefs.invalid || prefs.pristine">Save preferences</button></div>
      </form>
    </mat-tab>

    <mat-tab label="Skills ({{ skills().length }})">
      <div class="toolbar-row">
        <mat-form-field appearance="outline" class="grow"><mat-label>Add a skill</mat-label>
          <input matInput #skillInput (keydown.enter)="addSkill(skillInput.value); skillInput.value=''"></mat-form-field>
        <button mat-stroked-button (click)="addSkill(skillInput.value); skillInput.value=''">Add</button>
      </div>
      <mat-chip-set aria-label="Skills">
        @for (s of skills(); track s.id) {
          <mat-chip [highlighted]="s.verified" (removed)="removeSkill(s)" [matTooltip]="s.verified ? 'Verified – used as evidence' : 'Not verified – click to verify'"
            (click)="toggleVerified(s)">
            {{ s.name }} @if (!s.verified) { <mat-icon matChipTrailingIcon>help_outline</mat-icon> }
            <button matChipRemove [attr.aria-label]="'Remove ' + s.name"><mat-icon>cancel</mat-icon></button>
          </mat-chip>
        } @empty { <jp-empty icon="psychology" title="No skills yet" text="Import your resume or add skills manually." /> }
      </mat-chip-set>
    </mat-tab>

    @for (sec of sections; track sec.key) {
      <mat-tab [label]="sec.label + ' (' + items[sec.key]().length + ')'">
        <div class="toolbar-row"><span class="grow"></span>
          <button mat-flat-button color="primary" (click)="edit(sec.key)"><mat-icon>add</mat-icon> Add</button></div>
        @for (item of items[sec.key](); track item.id) {
          <mat-card class="item-card">
            <div class="row">
              <div class="grow">
                <div class="strong">{{ heading(sec.key, item) }} @if (!item.verified) { <span class="badge">unverified</span> }</div>
                <div class="muted">{{ sub(sec.key, item) }}</div>
                @if (item.achievements?.length || item.highlights?.length) {
                  <ul>@for (a of (item.achievements || item.highlights); track $index) { <li>{{ a }}</li> }</ul>
                }
              </div>
              <button mat-icon-button (click)="edit(sec.key, item)" aria-label="Edit"><mat-icon>edit</mat-icon></button>
              <button mat-icon-button (click)="remove(sec.key, item)" aria-label="Delete"><mat-icon>delete</mat-icon></button>
            </div>
          </mat-card>
        } @empty { <jp-empty [title]="'No ' + sec.label.toLowerCase() + ' yet'" /> }
      </mat-tab>
    }
  </mat-tab-group>`,
})
export class ProfileComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private dialog = inject(MatDialog);
  private fb = inject(FormBuilder);
  loading = signal(true);
  profile = signal<Profile | null>(null);
  skills = signal<CandidateSkill[]>([]);
  items: Record<Section, ReturnType<typeof signal<any[]>>> = {
    employment: signal<Employment[]>([]), education: signal<Education[]>([]), projects: signal<Project[]>([]), certifications: signal<Certification[]>([]),
  };
  sections: { key: Section; label: string }[] = [
    { key: 'employment', label: 'Experience' }, { key: 'education', label: 'Education' },
    { key: 'projects', label: 'Projects' }, { key: 'certifications', label: 'Certifications' },
  ];

  details = this.fb.group({
    full_name: [''], headline: [''], current_title: [''], total_experience_years: [0 as number | string, [Validators.min(0), Validators.max(60)]],
    email: ['', Validators.email], phone: [''], location: [''], linkedin_url: [''], github_url: [''], portfolio_url: [''], summary: [''],
    data_retention_days: [0, Validators.min(0)],
  });
  prefs = this.fb.group({
    target_roles: [''], preferred_locations: [''], company_types: [''], excluded_companies: [''], excluded_keywords: [''],
    work_models: [[] as string[]], employment_types: [[] as string[]], salary_currency: ['INR'],
    salary_min: [null as number | null, Validators.min(0)], salary_max: [null as number | null, Validators.min(0)],
    min_match_score: [60, [Validators.min(0), Validators.max(100)]], open_to_relocation: [false],
  });

  ngOnInit(): void { this.load(); }

  load(): void {
    this.loading.set(true);
    const list = <T>(path: string) => this.api.get<Paginated<T>>(path, { page_size: 100 });
    forkJoin({
      profile: this.api.get<Profile>('profile/'), skills: list<CandidateSkill>('profile/skills/'),
      employment: list<Employment>('profile/employment/'), education: list<Education>('profile/education/'),
      projects: list<Project>('profile/projects/'), certifications: list<Certification>('profile/certifications/'),
    }).subscribe({
      next: r => {
        this.profile.set(r.profile);
        this.skills.set(r.skills.results);
        (['employment', 'education', 'projects', 'certifications'] as Section[]).forEach(k => this.items[k].set((r as any)[k].results));
        this.details.reset({ ...r.profile } as any);
        const p = r.profile.preferences;
        const lines = (v: string[]) => v.join('\n');
        this.prefs.reset({
          ...p, target_roles: lines(p.target_roles), preferred_locations: lines(p.preferred_locations), company_types: lines(p.company_types),
          excluded_companies: lines(p.excluded_companies), excluded_keywords: lines(p.excluded_keywords),
        } as any);
        this.loading.set(false);
      },
      error: e => { this.loading.set(false); this.notify.error(e); },
    });
  }

  checkList(p: Profile) { return Object.entries(p.completeness.checks).map(([key, ok]) => ({ key, ok })); }
  lpa(v: number | null | undefined): string { return v ? `₹${(v / 100000).toFixed(1)} LPA` : ''; }

  saveDetails(): void {
    this.api.patch<Profile>('profile/', this.details.value).subscribe({
      next: p => { this.profile.set(p); this.details.markAsPristine(); this.notify.success('Profile saved. Matches will refresh.'); this.rematch(); },
      error: e => this.notify.error(e),
    });
  }

  savePrefs(): void {
    const v = this.prefs.value;
    const split = (s: string | null | undefined) => (s || '').split('\n').map(x => x.trim()).filter(Boolean);
    const body = { ...v, target_roles: split(v.target_roles), preferred_locations: split(v.preferred_locations), company_types: split(v.company_types),
      excluded_companies: split(v.excluded_companies), excluded_keywords: split(v.excluded_keywords) };
    this.api.patch('profile/preferences/', body).subscribe({
      next: () => { this.prefs.markAsPristine(); this.notify.success('Preferences saved. Recomputing match scores…'); this.rematch(); },
      error: e => this.notify.error(e),
    });
  }

  private rematch(): void { this.api.post('matches/rematch/').subscribe({ error: () => undefined }); }

  addSkill(name: string): void {
    if (!name.trim()) return;
    this.api.post<CandidateSkill>('profile/skills/', { skill_name: name.trim(), verified: true }).subscribe({
      next: s => this.skills.update(list => [...list, s]), error: e => this.notify.error(e) });
  }
  toggleVerified(s: CandidateSkill): void {
    this.api.patch<CandidateSkill>(`profile/skills/${s.id}/`, { verified: !s.verified }).subscribe({
      next: u => this.skills.update(list => list.map(x => (x.id === u.id ? u : x))), error: e => this.notify.error(e) });
  }
  removeSkill(s: CandidateSkill): void {
    this.api.delete(`profile/skills/${s.id}/`).subscribe({
      next: () => this.skills.update(list => list.filter(x => x.id !== s.id)), error: e => this.notify.error(e) });
  }

  heading(sec: Section, i: any): string {
    return sec === 'employment' ? `${i.title} — ${i.company}` : sec === 'education' ? (i.degree || i.institution) : i.name;
  }
  sub(sec: Section, i: any): string {
    if (sec === 'employment') return [i.location, `${i.start_date ?? '?'} – ${i.is_current ? 'Present' : (i.end_date ?? '?')}`].filter(Boolean).join(' · ');
    if (sec === 'education') return [i.institution, i.end_year, i.grade].filter(Boolean).join(' · ');
    if (sec === 'projects') return (i.technologies || []).join(', ');
    return [i.issuer, i.issued_on].filter(Boolean).join(' · ');
  }

  edit(sec: Section, item?: any): void {
    const title = `${item ? 'Edit' : 'Add'} ${this.sections.find(s => s.key === sec)!.label.toLowerCase()}`;
    this.dialog.open(FormDialogComponent, { width: '640px', maxWidth: '95vw', data: { title, fields: FIELDS[sec], value: item ?? { verified: true } } })
      .afterClosed().subscribe(value => {
        if (!value) return;
        const req = item ? this.api.patch(`profile/${sec}/${item.id}/`, value) : this.api.post(`profile/${sec}/`, value);
        req.subscribe({ next: () => { this.notify.success('Saved.'); this.load(); }, error: e => this.notify.error(e) });
      });
  }

  remove(sec: Section, item: any): void {
    this.dialog.open(ConfirmDialogComponent, { data: { title: 'Delete entry?', message: this.heading(sec, item), confirm: 'Delete', danger: true } })
      .afterClosed().subscribe(ok => ok && this.api.delete(`profile/${sec}/${item.id}/`).subscribe({
        next: () => this.items[sec].update(list => list.filter((x: any) => x.id !== item.id)), error: e => this.notify.error(e) }));
  }
}
