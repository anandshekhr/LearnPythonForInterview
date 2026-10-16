import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatDialog } from '@angular/material/dialog';
import { filter, interval, switchMap } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Paginated, Resume, ResumeVersion } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { ConfirmDialogComponent, EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Resume manager" subtitle="Your master resume is never modified automatically.">
    <input #file type="file" hidden accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      (change)="upload(file.files); file.value=''">
    <button mat-flat-button color="primary" (click)="file.click()" [disabled]="uploading()"><mat-icon>upload</mat-icon> Upload PDF / DOCX</button>
  </jp-page-header>
  <jp-loading [loading]="loading() || uploading()" />
  <div class="dropzone" (dragover)="$event.preventDefault()" (drop)="$event.preventDefault(); upload($event.dataTransfer?.files ?? null)">
    <mat-icon>cloud_upload</mat-icon> Drop a resume here (PDF or DOCX, max 5 MB). It is stored encrypted and parsed in the background.
  </div>
  <h2 class="section-title">Uploaded resumes</h2>
  @for (r of resumes(); track r.id) {
    <mat-card class="item-card">
      <div class="row">
        <mat-icon>{{ r.content_type.includes('pdf') ? 'picture_as_pdf' : 'description' }}</mat-icon>
        <div class="grow">
          <div class="strong">{{ r.title }} @if (r.is_master) { <span class="badge ok">master</span> }</div>
          <div class="muted">{{ r.original_filename }} · {{ (r.size_bytes / 1024).toFixed(0) }} KB · {{ r.created_at | date: 'medium' }}</div>
          <div>
            @switch (r.parse_status) {
              @case ('pending') { <span class="badge">queued</span> }
              @case ('parsing') { <span class="badge">parsing…</span> }
              @case ('parsed') { <span class="badge warn">ready to review</span> }
              @case ('applied') { <span class="badge ok">applied to profile</span> }
              @case ('failed') { <span class="badge danger">failed</span> <span class="error-text">{{ r.parse_error }}</span> }
            }
          </div>
        </div>
        @if (r.parse_status === 'parsed' || r.parse_status === 'applied') {
          <a mat-flat-button color="primary" [routerLink]="['/resumes', r.id, 'review']">Review & verify</a>
        }
        @if (r.parse_status === 'failed') { <button mat-stroked-button (click)="reparse(r)">Retry</button> }
        <button mat-icon-button (click)="api.download('resumes/' + r.id + '/download/', r.original_filename)" aria-label="Download"><mat-icon>download</mat-icon></button>
        <button mat-icon-button (click)="remove(r)" aria-label="Delete"><mat-icon>delete</mat-icon></button>
      </div>
    </mat-card>
  } @empty { <jp-empty icon="upload_file" title="No resumes yet" text="Upload your master resume to build your verified profile." /> }

  <h2 class="section-title">Resume versions</h2>
  <table mat-table [dataSource]="versions()" class="full">
    <ng-container matColumnDef="v"><th mat-header-cell *matHeaderCellDef>Version</th><td mat-cell *matCellDef="let v">v{{ v.version_number }}</td></ng-container>
    <ng-container matColumnDef="label"><th mat-header-cell *matHeaderCellDef>Label</th><td mat-cell *matCellDef="let v">{{ v.label }}</td></ng-container>
    <ng-container matColumnDef="kind"><th mat-header-cell *matHeaderCellDef>Kind</th><td mat-cell *matCellDef="let v">{{ v.kind }}</td></ng-container>
    <ng-container matColumnDef="status"><th mat-header-cell *matHeaderCellDef>Status</th><td mat-cell *matCellDef="let v"><span class="badge">{{ v.status.replace('_', ' ') }}</span></td></ng-container>
    <ng-container matColumnDef="created"><th mat-header-cell *matHeaderCellDef>Created</th><td mat-cell *matCellDef="let v">{{ v.created_at | date: 'mediumDate' }}</td></ng-container>
    <tr mat-header-row *matHeaderRowDef="cols"></tr>
    <tr mat-row *matRowDef="let v; columns: cols" class="clickable" [routerLink]="['/resume-versions', v.id]"></tr>
  </table>
  @if (!versions().length) { <jp-empty icon="history" title="No versions yet" text="A master snapshot is created when you apply a parsed resume." /> }
  <div class="toolbar-row"><button mat-stroked-button (click)="snapshot()"><mat-icon>photo_camera</mat-icon> Snapshot current profile</button></div>`,
})
export class ResumesComponent implements OnInit {
  api = inject(ApiService);
  private notify = inject(NotifyService);
  private dialog = inject(MatDialog);
  private destroyRef = inject(DestroyRef);
  loading = signal(true);
  uploading = signal(false);
  resumes = signal<Resume[]>([]);
  versions = signal<ResumeVersion[]>([]);
  cols = ['v', 'label', 'kind', 'status', 'created'];

  ngOnInit(): void {
    this.load();
    interval(3000).pipe(filter(() => this.resumes().some(r => ['pending', 'parsing'].includes(r.parse_status))),
      switchMap(() => this.api.get<Paginated<Resume>>('resumes/')), takeUntilDestroyed(this.destroyRef))
      .subscribe(r => this.resumes.set(r.results));
  }

  load(): void {
    this.api.get<Paginated<Resume>>('resumes/').subscribe({ next: r => { this.resumes.set(r.results); this.loading.set(false); },
      error: e => { this.loading.set(false); this.notify.error(e); } });
    this.api.get<Paginated<ResumeVersion>>('resume-versions/', { page_size: 50 }).subscribe(r => this.versions.set(r.results));
  }

  upload(files: FileList | null): void {
    const file = files?.[0];
    if (!file) return;
    if (!/\.(pdf|docx)$/i.test(file.name)) { this.notify.error(null, 'Only PDF and DOCX files are supported.'); return; }
    if (file.size > 5 * 1024 * 1024) { this.notify.error(null, 'File is larger than 5 MB.'); return; }
    const form = new FormData();
    form.append('file', file);
    form.append('title', file.name.replace(/\.(pdf|docx)$/i, ''));
    this.uploading.set(true);
    this.api.post<Resume>('resumes/', form).subscribe({
      next: r => { this.uploading.set(false); this.resumes.update(l => [r, ...l]); this.notify.success('Uploaded. Parsing in the background…'); },
      error: e => { this.uploading.set(false); this.notify.error(e, 'Upload failed.'); },
    });
  }

  reparse(r: Resume): void {
    this.api.post<Resume>(`resumes/${r.id}/parse/`).subscribe({ next: u => this.resumes.update(l => l.map(x => x.id === u.id ? u : x)),
      error: e => this.notify.error(e) });
  }

  remove(r: Resume): void {
    this.dialog.open(ConfirmDialogComponent, { data: { title: 'Delete resume?', message: `${r.original_filename} will be permanently deleted. Your verified profile is not affected.`, confirm: 'Delete', danger: true } })
      .afterClosed().subscribe(ok => ok && this.api.delete(`resumes/${r.id}/`).subscribe({
        next: () => this.resumes.update(l => l.filter(x => x.id !== r.id)), error: e => this.notify.error(e) }));
  }

  snapshot(): void {
    this.api.post('resume-versions/snapshot/').subscribe({ next: () => { this.notify.success('Snapshot created.'); this.load(); }, error: e => this.notify.error(e) });
  }
}
