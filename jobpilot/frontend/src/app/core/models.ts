export interface Paginated<T> { count: number; next: string | null; previous: string | null; results: T[]; }

export interface User { id: number; email: string; full_name: string; date_joined: string; }

export interface Preferences {
  target_roles: string[]; preferred_locations: string[]; work_models: string[]; employment_types: string[];
  industries: string[]; company_types: string[]; excluded_companies: string[]; excluded_keywords: string[];
  salary_currency: string; salary_min: number | null; salary_max: number | null; min_match_score: number;
  open_to_relocation: boolean; notice_period_days: number | null; updated_at?: string;
}

export interface Profile {
  id: number; full_name: string; headline: string; summary: string; current_title: string; location: string; email: string;
  phone: string; linkedin_url: string; github_url: string; portfolio_url: string; total_experience_years: string;
  verified_at: string | null; data_retention_days: number; preferences: Preferences;
  completeness: { score: number; checks: Record<string, boolean> };
}

export interface CandidateSkill { id: number; name: string; category: string; years: string | null; proficiency: string; is_primary: boolean; verified: boolean; source: string; }
export interface Employment { id: number; company: string; title: string; location: string; start_date: string | null; end_date: string | null; is_current: boolean; description: string; achievements: string[]; technologies: string[]; verified: boolean; }
export interface Education { id: number; institution: string; degree: string; field_of_study: string; start_year: number | null; end_year: number | null; grade: string; verified: boolean; }
export interface Project { id: number; name: string; role: string; description: string; highlights: string[]; technologies: string[]; url: string; verified: boolean; }
export interface Certification { id: number; name: string; issuer: string; issued_on: string | null; expires_on: string | null; credential_url: string; verified: boolean; }

export interface Resume {
  id: number; title: string; original_filename: string; content_type: string; size_bytes: number; is_master: boolean;
  parse_status: 'pending' | 'parsing' | 'parsed' | 'applied' | 'failed'; parse_error: string; parser: string;
  created_at: string; parsed_data?: ParsedResume;
}
export interface ParsedResume {
  contact: Record<string, string>; headline: string; summary: string; total_experience_years: number;
  stated_experience_years: number | null; skills: { name: string; category: string; in_skills_section: boolean }[];
  employment: Partial<Employment>[]; education: Partial<Education>[]; projects: Partial<Project>[];
  certifications: Partial<Certification>[]; warnings: string[]; parser: string;
}

export interface ResumeContent {
  contact: Record<string, string>; headline: string; summary: string; total_experience_years: number; skills: string[];
  employment: (Employment & { achievement_sources?: number[] })[]; education: Education[]; projects: Project[]; certifications: Certification[];
}
export interface RequirementRow { requirement: string; importance: 'required' | 'preferred'; status: 'matched' | 'evidence_only' | 'missing'; evidence: { type: string; label: string; text?: string }[]; }
export interface ResumeVersion {
  id: number; version_number: number; label: string; kind: 'master' | 'tailored'; status: string; job: number | null;
  job_title: string | null; company_name: string | null; base_version: number | null; content: ResumeContent;
  tailoring_notes: { requirement_map?: RequirementRow[]; warnings?: string[]; changes?: string[]; rewrites?: { original: string; rewritten: string }[] };
  generator: string; approved_at: string | null; created_at: string;
}
export interface CompareSection { section: string; heading?: string; master: any; tailored: any; changed: boolean; }
export interface Comparison { version_id: number; sections: CompareSection[]; requirement_map: RequirementRow[]; warnings: string[]; rewrites: { original: string; rewritten: string }[]; }

export interface GeneratedDocument {
  id: number; kind: 'resume' | 'cover_letter'; format: 'pdf' | 'docx' | 'text'; template: string; status: string; title: string;
  text_content: string; size_bytes: number; generator: string; warnings: string[]; job: number | null; job_title: string | null;
  company_name: string | null; resume_version: number | null; approved_at: string | null; created_at: string;
}

export interface Company { id: number; name: string; slug: string; website: string; careers_url: string; industry: string; company_type: string; headquarters: string; notes: string; source_count: number; active_jobs: number; can_edit: boolean; }
export interface Capabilities { list_jobs: boolean; job_details: boolean; salary_data: boolean; application_questions: boolean; api_submission: boolean; api_submission_requires: string; browser_assist: boolean; documentation_url: string; limitations: string[]; api_submission_active?: boolean; }
export interface JobSource {
  id: number; company: number; company_name: string; source_type: string; identifier: string; config: Record<string, string>;
  enabled: boolean; status: string; sync_interval_minutes: number; min_request_interval_seconds: number;
  last_synced_at: string | null; last_success_at: string | null; last_error: string; consecutive_failures: number;
  active_job_count: number; submission_enabled: boolean; has_submission_key: boolean; capabilities: Capabilities; can_edit: boolean;
}

export interface MatchSummary { id: number; score: number; category: string; user_state: string; }
export interface Job {
  id: number; title: string; company: number; company_name: string; source_type: string; department: string; location_raw: string;
  locations: string[]; work_model: string; employment_type: string; seniority: string; salary_min: number | null;
  salary_max: number | null; salary_currency: string; salary_text: string; published_at: string | null; first_seen_at: string;
  status: string; expired_at: string | null; official_url: string; duplicate_of: number | null; my_match: MatchSummary | null;
  application_id: number | null;
  description_text?: string; requirements?: { required_skills: string[]; preferred_skills: string[]; min_years: number | null; max_years: number | null; responsibilities: string[] };
  company_detail?: Partial<Company>; description_version?: number;
}
export interface Signal { score: number | null; weight: number; detail: string; }
export interface JobMatch {
  id: number; job: Job; score: number; category: string; components: Record<string, Signal>; skills_matched: string[];
  skills_missing: string[]; skills_unverified: string[]; relevant_experience: { kind: string; label: string; skills: string[] }[];
  gaps: string[]; location_fit: string; salary_fit: string; explanation: string; recommended_action: string;
  semantic_similarity: number | null; information_coverage: number; excluded_reason: string; user_state: string;
}
export interface ScoringConfig { [k: string]: number | string; }

export interface HistoryEntry { id: number; from_status: string; to_status: string; actor: string; note: string; created_at: string; }
export interface Answer { id: number; question_key: string; question_label: string; field_type: string; required: boolean; options: { label: string; value: any }[]; answer: string; origin: string; sensitive: boolean; reviewed_at: string | null; }
export interface Interview { id: number; application: number; company_name: string; job_title: string; round_name: string; scheduled_at: string; duration_minutes: number; mode: string; location: string; interviewers: string; notes: string; outcome: string; }
export interface FollowUp { id: number; application: number; company_name: string; job_title: string; due_date: string; note: string; completed_at: string | null; }
export interface Application {
  id: number; job: number | null; company_name: string; job_title: string; job_url: string; source_type: string;
  match_score: number | null; status: string; mode: string; resume_version: number | null; resume_version_label: string | null;
  cover_letter: number | null; submitted_at: string | null; submission_channel: string; submission_reference: string;
  follow_up_date: string | null; last_verified_at: string | null; next_interview: string | null; created_at: string; updated_at: string;
  notes?: string; approved_at?: string | null; submission_response?: any; submission_attempts?: number; last_error?: string;
  resume_document?: number | null; history?: HistoryEntry[]; answers?: Answer[]; interviews?: Interview[]; follow_ups?: FollowUp[];
  allowed_transitions?: string[];
}
export interface ApplicationKit {
  application_id: number; status: string; official_url: string; job_active: boolean; resume_document: number | null;
  cover_letter: number | null; answers: Answer[]; problems: string[]; mode: string;
  automated_submission: { available: boolean; source_supports_api: boolean; reason: string; consent_given: boolean };
  browser_assist_enabled: boolean;
}

export interface AutomationSettings {
  application_mode: 'discovery' | 'assisted' | 'automated'; automated_submission_consent_at: string | null;
  automated_submission_allowed: boolean; max_submissions_per_day: number; auto_shortlist_min_score: number | null;
  auto_prepare_documents_min_score: number | null; notify_min_score: number; notify_new_matches: boolean;
  notify_awaiting_review: boolean; notify_submissions: boolean; notify_interviews: boolean; notify_follow_ups: boolean;
  notify_expired_jobs: boolean; email_notifications: boolean;
}
export interface AutomationRun { id: number; kind: string; status: string; target_type: string; target_id: string; started_at: string; finished_at: string | null; duration_seconds: number | null; stats: any; error: string; }
export interface ScheduledSearch { id: number; name: string; enabled: boolean; interval_hours: number; sources: number[]; last_run_at: string | null; next_run_at: string; }
export interface AppNotification { id: number; kind: string; title: string; body: string; link: string; read_at: string | null; created_at: string; }

export interface Overview {
  jobs_discovered: number; active_jobs: number; new_jobs_this_week: number; high_match_jobs: number; shortlisted: number;
  applications_total: number; applications_submitted: number; submitted_this_week: number; awaiting_review: number;
  ready_to_apply: number; submission_failures: number; interviews_scheduled: number; offers_received: number; rejections: number;
  conversion: { response_rate: number | null; interview_rate: number | null; offer_rate: number | null; interview_to_offer: number | null };
}

export const STATUS_LABELS: Record<string, string> = {
  discovered: 'Discovered', shortlisted: 'Shortlisted', resume_prepared: 'Resume prepared', awaiting_approval: 'Awaiting approval',
  ready_to_apply: 'Ready to apply', submitting: 'Submitting', submitted: 'Submitted', submission_failed: 'Submission failed',
  assessment: 'Assessment', interview_scheduled: 'Interview scheduled', interview_completed: 'Interview completed',
  rejected: 'Rejected', offer_received: 'Offer received', withdrawn: 'Withdrawn', closed: 'Closed',
};
