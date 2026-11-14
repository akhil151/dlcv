import { createBrowserRouter, Navigate } from 'react-router-dom';
import { LandingPage } from '../pages/LandingPage';
import { DashboardPage } from '../pages/DashboardPage';
import { UploadPage } from '../pages/UploadPage';
import { ProcessingPage } from '../pages/ProcessingPage';
import { ResultsPage } from '../pages/ResultsPage';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <LandingPage />,
  },
  {
    path: '/dashboard',
    element: <DashboardPage />,
  },
  {
    path: '/upload',
    element: <UploadPage />,
  },
  {
    // Without a real task id there is nothing to process: no demo task exists.
    path: '/processing',
    element: <Navigate to="/upload" replace />,
  },
  {
    path: '/processing/:taskId',
    element: <ProcessingPage />,
  },
  {
    path: '/results',
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: '/results/:taskId',
    element: <ResultsPage />,
  },
]);
