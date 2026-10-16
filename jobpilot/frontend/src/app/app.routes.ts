import { Routes } from '@angular/router';
import { authGuard } from './core/auth.guard';
import { ShellComponent } from './layout/shell.component';
import { LoginComponent } from './pages/login.component';

const page = (title: string, load: () => Promise<any>, path: string) => ({ path, title: `${title} · JobPilot`, loadComponent: load });

export const routes: Routes = [
  { path: 'login', component: LoginComponent, title: 'Sign in · JobPilot' },
  {
    path: '', component: ShellComponent, canActivate: [authGuard], children: [
      { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
      page('Overview', () => import('./pages/dashboard.component').then(m => m.DashboardComponent), 'dashboard'),
      page('Career profile', () => import('./pages/profile.component').then(m => m.ProfileComponent), 'profile'),
      page('Resume manager', () => import('./pages/resumes.component').then(m => m.ResumesComponent), 'resumes'),
      page('Review resume', () => import('./pages/resume-review.component').then(m => m.ResumeReviewComponent), 'resumes/:id/review'),
      page('Job discovery', () => import('./pages/jobs.component').then(m => m.JobsComponent), 'jobs'),
      page('Job details', () => import('./pages/job-detail.component').then(m => m.JobDetailComponent), 'jobs/:id'),
      page('Recommended jobs', () => import('./pages/recommended.component').then(m => m.RecommendedComponent), 'recommended'),
      page('Tailored resumes', () => import('./pages/documents.component').then(m => m.DocumentsComponent), 'documents'),
      page('Resume version', () => import('./pages/resume-version.component').then(m => m.ResumeVersionComponent), 'resume-versions/:id'),
      page('Applications', () => import('./pages/applications.component').then(m => m.ApplicationsComponent), 'applications'),
      page('Application', () => import('./pages/application-detail.component').then(m => m.ApplicationDetailComponent), 'applications/:id'),
      page('Interviews', () => import('./pages/interviews.component').then(m => m.InterviewsComponent), 'interviews'),
      page('Companies & sources', () => import('./pages/sources.component').then(m => m.SourcesComponent), 'sources'),
      page('Automation settings', () => import('./pages/settings.component').then(m => m.SettingsComponent), 'settings'),
      page('Notifications', () => import('./pages/notifications.component').then(m => m.NotificationsComponent), 'notifications'),
      page('Analytics', () => import('./pages/analytics.component').then(m => m.AnalyticsComponent), 'analytics'),
    ],
  },
  { path: '**', redirectTo: '' },
];
