import { DatePipe, NgClass } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatBadgeModule } from '@angular/material/badge';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatCardModule } from '@angular/material/card';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatChipsModule } from '@angular/material/chips';
import { MatDialogModule } from '@angular/material/dialog';
import { MatDividerModule } from '@angular/material/divider';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSliderModule } from '@angular/material/slider';
import { MatSortModule } from '@angular/material/sort';
import { MatTableModule } from '@angular/material/table';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTooltipModule } from '@angular/material/tooltip';
import { RouterLink } from '@angular/router';

export const MATERIAL = [
  DatePipe, NgClass, FormsModule, ReactiveFormsModule, RouterLink, MatAutocompleteModule, MatBadgeModule, MatButtonModule,
  MatButtonToggleModule, MatCardModule, MatCheckboxModule, MatChipsModule, MatDialogModule, MatDividerModule,
  MatExpansionModule, MatFormFieldModule, MatIconModule, MatInputModule, MatListModule, MatMenuModule, MatPaginatorModule,
  MatProgressBarModule, MatProgressSpinnerModule, MatSelectModule, MatSlideToggleModule, MatSliderModule, MatSortModule,
  MatTableModule, MatTabsModule, MatTooltipModule,
];
