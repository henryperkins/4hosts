import React from 'react';
import { HostParadigm } from '@/types/paradigm';

interface ParadigmCardProps {
  paradigm: HostParadigm;
  title: string;
  description: string;
  isActive?: boolean;
  onClick?: () => void;
}

const paradigmStyles = {
  dolores: {
    container: 'bg-red-50 border-red-200 hover:border-red-400',
    title: 'text-red-900',
    description: 'text-red-700',
    icon: '✊'
  },
  teddy: {
    container: 'bg-blue-50 border-blue-200 hover:border-blue-400',
    title: 'text-blue-900',
    description: 'text-blue-700',
    icon: '❤️'
  },
  bernard: {
    container: 'bg-gray-50 border-gray-200 hover:border-gray-400',
    title: 'text-gray-900',
    description: 'text-gray-700',
    icon: '📊'
  },
  maeve: {
    container: 'bg-purple-50 border-purple-200 hover:border-purple-400',
    title: 'text-purple-900',
    description: 'text-purple-700',
    icon: '♟️'
  }
};

export const ParadigmCard: React.FC<ParadigmCardProps> = ({
  paradigm,
  title,
  description,
  isActive = false,
  onClick
}) => {
  const styles = paradigmStyles[paradigm];

  return (
    <div
      className={`
        relative p-6 rounded-lg border-2 cursor-pointer transition-all
        ${styles.container}
        ${isActive ? 'ring-2 ring-offset-2 ring-' + paradigm : ''}
      `}
      onClick={onClick}
      role="button"
      tabIndex={0}
      aria-label={`Select ${paradigm} paradigm`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick?.();
        }
      }}
    >
      <div className="flex items-start space-x-3">
        <span className="text-2xl" role="img" aria-label={`${paradigm} icon`}>
          {styles.icon}
        </span>
        <div className="flex-1">
          <h3 className={`text-lg font-semibold ${styles.title}`}>
            {title}
          </h3>
          <p className={`mt-1 text-sm ${styles.description}`}>
            {description}
          </p>
        </div>
      </div>
      {isActive && (
        <div className="absolute top-2 right-2">
          <span className="text-xs font-medium text-green-600">Active</span>
        </div>
      )}
    </div>
  );
};

// Example usage:
/*
<ParadigmCard
  paradigm="dolores"
  title="Revolutionary Research"
  description="Expose systemic issues and uncover hidden truths"
  isActive={selectedParadigm === 'dolores'}
  onClick={() => setSelectedParadigm('dolores')}
/>
*/